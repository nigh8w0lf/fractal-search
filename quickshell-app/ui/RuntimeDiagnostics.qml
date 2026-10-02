import QtQuick
import Quickshell.Io

// A separate CLI invocation stays available when CUDA prevents worker startup.
Item {
    id: diagnostics
    required property string executable
    required property string provider
    required property int deviceId
    required property string libraryDirectory
    property bool checking: false
    property var report: null
    property string errorMessage: ""
    property string processError: ""
    readonly property string summary: checking ? "Checking accelerator requirements…" : errorMessage.length ? errorMessage : report ? report.summary : "Check the NVIDIA driver, CUDA runtime and ONNX Runtime provider requirements."
    readonly property string details: report ? formatReport(report) : ""
    signal failed(string message)

    function fail(message) {
        errorMessage = String(message);
        failed(errorMessage);
    }
    function refresh() {
        if (checking)
            return;
        if (!executable.startsWith("/")) {
            fail("The diagnostics executable is not configured. Reinstall the plugin or set FRACTAL_SEARCH_BINARY in the developer harness.");
            return;
        }
        if (provider !== "cpu" && provider !== "cuda") {
            fail("The configured inference provider is invalid. Reinstall the plugin.");
            return;
        }
        report = null;
        errorMessage = "";
        processError = "";
        process.command = [executable, "diagnostics", "--provider", provider];
        if (provider === "cuda")
            process.command = process.command.concat(["--device-id", String(deviceId)]);
        // Installed bundles provide the exact service loader path, including
        // ORT's provider directory. Developer harnesses retain their environment.
        process.environment = libraryDirectory.length ? {
            LD_LIBRARY_PATH: libraryDirectory
        } : {};
        checking = true;
        timeout.start();
        process.running = true;
    }
    function formatReport(value) {
        const lines = ["Fractal Search CUDA diagnostics", "Provider: " + value.selectedProvider + (value.selectedDeviceId !== null ? " · device " + value.selectedDeviceId : ""), "CUDA-enabled build: " + (value.cudaEnabledBuild ? "yes" : "no"), "Platform: " + value.platform, "", value.summary, "", "NVIDIA driver: " + value.nvidiaDriver.state + " — " + value.nvidiaDriver.detail];
        value.gpus.forEach(gpu => lines.push("GPU " + gpu.index + ": " + gpu.name + " · driver " + gpu.driverVersion));
        lines.push("", "Optional CUDA toolkit: " + value.cudaToolkit.detail, "", "Linker cache: " + value.linkerCache.detail, "CUDA provider: " + value.providerDependencies.detail, "", "Required libraries:");
        value.libraries.forEach(library => {
            const state = library.state === "notFound" ? "NOT FOUND in checked locations" : library.state === "notVisible" ? "INSTALLED, outside loader path" : "Found in loader/provider path";
            lines.push(library.name + " (" + library.component + "): " + state);
            library.paths.forEach(path => lines.push("  " + path));
        });
        if (value.probeErrors.length) {
            lines.push("", "Incomplete checks:");
            value.probeErrors.forEach(error => lines.push(error));
        }
        lines.push("", "Setup steps:");
        value.installSteps.forEach((step, index) => lines.push((index + 1) + ". " + step));
        lines.push("", "LD_LIBRARY_PATH: " + (value.ldLibraryPath || "(empty)"), "CUDA_VISIBLE_DEVICES: " + (value.cudaVisibleDevices === null ? "(unset)" : value.cudaVisibleDevices), "", "Checked directories:");
        value.librarySearchLocations.forEach(path => lines.push(path));
        return lines.join("\n");
    }
    Process {
        id: process
        stdout: SplitParser {
            onRead: data => {
                try {
                    if (diagnostics.report !== null || data.length > 65536)
                        throw new Error("Invalid or oversized accelerator diagnostic response.");
                    const value = JSON.parse(data);
                    if (value.schemaVersion !== 1 || typeof value.checksPassed !== "boolean" || typeof value.summary !== "string" || !Array.isArray(value.libraries) || !Array.isArray(value.gpus) || !Array.isArray(value.probeErrors) || !Array.isArray(value.installSteps) || !Array.isArray(value.librarySearchLocations) || !value.nvidiaDriver || !value.cudaToolkit || !value.linkerCache || !value.providerDependencies)
                        throw new Error("Incompatible accelerator diagnostic response. Reinstall the plugin.");
                    diagnostics.report = value;
                } catch (error) {
                    diagnostics.fail(error.message);
                }
            }
        }
        stderr: SplitParser {
            onRead: data => diagnostics.processError = (diagnostics.processError + "\n" + data).slice(-4096)
        }
        // qmllint disable signal-handler-parameters
        onExited: (exitCode, exitStatus) => {
            timeout.stop();
            diagnostics.checking = false;
            if (exitCode !== 0 || exitStatus !== 0 || !diagnostics.report)
                diagnostics.fail(diagnostics.errorMessage || "Accelerator diagnostics failed (exit " + exitCode + "). " + diagnostics.processError.trim());
        }
        // qmllint enable signal-handler-parameters
        onRunningChanged: {
            if (!running && diagnostics.checking) {
                timeout.stop();
                diagnostics.checking = false;
                diagnostics.fail("Cannot start accelerator diagnostics. Check the executable: " + diagnostics.executable + ". " + diagnostics.processError.trim());
            }
        }
    }
    Timer {
        id: timeout
        interval: 20000
        onTriggered: {
            diagnostics.checking = false;
            diagnostics.fail("Accelerator diagnostics exceeded 20 seconds. Check NVIDIA driver utilities and library access, then refresh.");
            process.running = false;
        }
    }
}
