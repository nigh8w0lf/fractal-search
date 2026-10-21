import QtQuick
import Quickshell

QtObject {
    readonly property string mcpExecutable: Quickshell.env("FRACTAL_SEARCH_MCP")
    readonly property string socketPath: Quickshell.env("XDG_RUNTIME_DIR") + "/fractal-search/backend.sock"
    readonly property string bundleVersion: "development-harness"
    readonly property string serviceName: "fractal-search.service"
    readonly property string backendExecutable: Quickshell.env("FRACTAL_SEARCH_BINARY")
    readonly property string configuredProvider: Quickshell.env("FRACTAL_SEARCH_PROVIDER") || "cpu"
    readonly property int configuredDeviceId: Quickshell.env("FRACTAL_SEARCH_DEVICE_ID") ? Number(Quickshell.env("FRACTAL_SEARCH_DEVICE_ID")) : configuredProvider === "cuda" ? 0 : -1
    readonly property string runtimeLibraryPath: ""
    readonly property string cudaLibraryDirectory: Quickshell.env("FRACTAL_SEARCH_CUDA_LIB_DIR")
    readonly property string pickerExecutable: Quickshell.env("FRACTAL_SEARCH_PICKER")
    readonly property bool closeShortcutEnabled: Quickshell.env("FRACTAL_SEARCH_CLOSE_SHORTCUT") === "1"
}
