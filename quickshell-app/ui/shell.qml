// Isolated developer harness only. The installed plugin never creates ShellRoot.
import QtQuick
import Quickshell
import Quickshell.Io

ShellRoot {
    SearchPanel {
        id: panel
        manageService: false
        socketPath: Quickshell.env("FRACTAL_SEARCH_SOCKET")
        onStopped: Qt.quit()
        Component.onCompleted: open("{}")
    }
    IpcHandler {
        target: "search"
        function summon(): void {
            panel.open("{}");
        }
        function toggle(): void {
            if (panel.opened)
                panel.close();
            else
                panel.open("{}");
        }
        function query(text: string): void {
            panel.query(text);
        }
        function close(): void {
            panel.quit();
        }
    }
}
