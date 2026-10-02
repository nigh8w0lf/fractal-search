import QtQuick
import Quickshell.Io

Item {
    id: control
    required property string serviceName
    property bool enabledForSession: true
    property bool startAtLogin: false
    property bool known: false
    readonly property bool busy: reader.running || writer.running
    signal failed(string message)

    function refresh() {
        if (enabledForSession && !busy)
            reader.running = true;
    }
    function setStartAtLogin(enabled) {
        if (!enabledForSession || !known || busy)
            return;
        writer.command = ["systemctl", "--user", enabled ? "enable" : "disable", serviceName];
        writer.running = true;
    }
    Process {
        id: reader
        command: ["systemctl", "--user", "is-enabled", control.serviceName]
        stdout: StdioCollector {
            id: stateOutput
            waitForEnd: true
        }
        stderr: StdioCollector {
            id: stateError
            waitForEnd: true
        }
        // qmllint disable signal-handler-parameters
        onExited: exitCode => {
            const state = stateOutput.text.trim();
            if (exitCode === 0 && state === "enabled") {
                control.startAtLogin = true;
                control.known = true;
            } else if (exitCode === 1 && state === "disabled") {
                control.startAtLogin = false;
                control.known = true;
            } else {
                control.known = false;
                control.failed("Cannot read Load search model at login for " + control.serviceName + ": " + (stateError.text.trim() || state || "systemctl exited with code " + exitCode));
            }
        }
        // qmllint enable signal-handler-parameters
    }
    Process {
        id: writer
        stderr: StdioCollector {
            id: changeError
            waitForEnd: true
        }
        // qmllint disable signal-handler-parameters
        onExited: exitCode => {
            if (exitCode !== 0)
                control.failed("Cannot change Load search model at login for " + control.serviceName + ": " + (changeError.text.trim() || "systemctl exited with code " + exitCode));
            // Read the actual unit state even after a failed or partial change.
            reader.running = true;
        }
        // qmllint enable signal-handler-parameters
    }
}
