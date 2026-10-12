import QtQuick
import Quickshell.Io

QtObject {
    id: opener
    property bool opening: false
    property string diagnostic: ""
    signal opened
    signal failed(string message)

    function open(url) {
        if (opening) {
            failed("A file is already being opened. Wait before trying again.");
            return;
        }
        if (typeof url !== "string" || !url.startsWith("file:///")) {
            failed("Cannot open this file: the service returned an invalid local file URL.");
            return;
        }
        diagnostic = "";
        opening = true;
        timeout.start();
        // GIO resolves local MIME types and honors desktop-entry Terminal=true.
        // xdg-open's generic path can run terminal editors without a terminal
        // or substitute a browser when content sniffing finds no handler.
        process.command = ["gio", "open", url];
        process.running = true;
    }

    property Process process: Process {
        id: process
        stderr: SplitParser {
            onRead: data => opener.diagnostic = (opener.diagnostic + "\n" + data).slice(-4096)
        }
        // qmllint disable signal-handler-parameters
        onExited: (exitCode, exitStatus) => {
            timeout.stop();
            if (!opener.opening)
                return;
            opener.opening = false;
            if (exitCode !== 0 || exitStatus !== 0)
                opener.failed("The desktop could not open this file (exit " + exitCode + "). " + opener.diagnostic.trim());
            else
                opener.opened();
        }
        // qmllint enable signal-handler-parameters
        onRunningChanged: {
            // Quickshell does not emit exited for FailedToStart.
            if (!running && opener.opening) {
                timeout.stop();
                opener.opening = false;
                opener.failed("Cannot start the desktop file opener. Install GLib's gio command and try again.");
            }
        }
    }
    property Timer timeout: Timer {
        id: timeout
        interval: 15000
        onTriggered: {
            opener.opening = false;
            opener.failed("The desktop file opener did not respond within 15 seconds. Check the default application for this file type.");
            process.running = false;
        }
    }
}
