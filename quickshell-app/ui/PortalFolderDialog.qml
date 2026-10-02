import QtQuick
import Quickshell.Io

QtObject {
    id: picker
    required property string executable
    property string title: "Choose a folder"
    property string selectedFolder: ""
    property bool visible: false
    property bool cancelRequested: false
    property var response: null
    property string diagnostic: ""
    signal accepted
    signal rejected
    signal failed(string message)

    function open() {
        if (process.running) {
            failed("A folder chooser is already open. Close it before opening another.");
            return;
        }
        if (!executable.startsWith("/")) {
            failed("The native folder chooser executable is not configured. Reinstall the plugin.");
            return;
        }
        response = null;
        diagnostic = "";
        cancelRequested = false;
        visible = true;
        process.command = [executable, "--title", title];
        process.running = true;
    }
    function close() {
        cancelRequested = true;
        // SIGTERM asks the helper to close its portal request. Destruction also
        // disconnects its private bus client, so reload cannot strand a picker.
        process.running = false;
        visible = false;
    }
    Component.onDestruction: close()

    property Process process: Process {
        id: process
        stdout: SplitParser {
            onRead: data => {
                try {
                    if (picker.response !== null || data.length > 65536)
                        throw new Error("Invalid folder chooser response.");
                    const result = JSON.parse(data);
                    if (typeof result.ok !== "boolean" || (result.ok && typeof result.cancelled !== "boolean") || (result.ok && !result.cancelled && typeof result.folder !== "string") || (!result.ok && typeof result.error !== "string"))
                        throw new Error("Invalid folder chooser response.");
                    picker.response = result;
                } catch (error) {
                    picker.diagnostic = error.message;
                    picker.response = {
                        ok: false,
                        error: error.message
                    };
                }
            }
        }
        stderr: SplitParser {
            onRead: data => {
                picker.diagnostic = data.slice(0, 2048);
                console.debug("[Fractal Search picker] " + data);
            }
        }
        // qmllint disable signal-handler-parameters
        onExited: (exitCode, exitStatus) => {
            picker.visible = false;
            const result = picker.response;
            if (exitCode !== 0 || exitStatus !== 0 || !result || !result.ok) {
                picker.failed(result && !result.ok ? result.error : "The native folder chooser failed (exit " + exitCode + "). " + picker.diagnostic);
            } else if (picker.cancelRequested || result.cancelled) {
                picker.rejected();
            } else {
                picker.selectedFolder = result.folder;
                picker.accepted();
            }
        }
        // qmllint enable signal-handler-parameters
        onRunningChanged: {
            // FailedToStart does not emit exited in Quickshell.
            if (!running && picker.visible) {
                picker.visible = false;
                picker.failed("Cannot start the native folder chooser. Reinstall the plugin and check its executable: " + picker.executable);
            }
        }
    }
}
