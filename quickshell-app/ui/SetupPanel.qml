import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Quickshell
import Quickshell.Io
import Quickshell.Wayland
import qs.Commons
import qs.Commons as Omarchy
import qs.Ui as OmarchyUi

Item {
    id: setup
    property var shell: null
    property var manifest: null
    readonly property bool opened: window.visible
    property string stage: "idle"
    property string message: "Choose a backend to get started."
    property string errorMessage: ""
    readonly property bool busy: ["starting", "downloading", "verifying", "installing"].indexOf(stage) >= 0 || launch.running
    readonly property string helper: decodeURIComponent(Qt.resolvedUrl("../packaging/first-run.sh").toString().replace(/^file:\/\//, ""))
    readonly property string sourceDirectory: decodeURIComponent(Qt.resolvedUrl("../../").toString().replace(/^file:\/\//, "")).replace(/\/$/, "")
    readonly property var popupColors: Omarchy.Color.popups
    readonly property var colors: ({background: popupColors.background, foreground: popupColors.text, accent: Omarchy.Color.accent, error: Omarchy.Color.urgent})

    function open(payload) { window.visible = true; refresh(); }
    function close() { window.visible = false; }
    function quit() { close(); }
    function ping() { return "first-run-setup"; }
    function refresh() { if (!poll.running) poll.running = true; }
    function install(provider) {
        if (busy) return;
        errorMessage = "";
        launch.command = ["bash", helper, "launch", provider, sourceDirectory, provider === "cuda" ? cudaDirectory.text.trim() : ""];
        launch.running = true;
    }
    Process {
        id: launch
        stderr: StdioCollector { id: launchErrors; waitForEnd: true }
        // qmllint disable signal-handler-parameters
        onExited: (code, status) => {
            if (code !== 0 || status !== 0)
                setup.errorMessage = "Cannot start installation. " + launchErrors.text.slice(-6000);
            setup.refresh();
        }
        // qmllint enable signal-handler-parameters
    }
    Process {
        id: poll
        command: ["bash", setup.helper, "status"]
        stdout: StdioCollector { id: stateOutput; waitForEnd: true }
        stderr: StdioCollector { id: stateErrors; waitForEnd: true }
        // qmllint disable signal-handler-parameters
        onExited: (code, status) => {
            if (code !== 0 || status !== 0) {
                setup.errorMessage = "Cannot read setup status. " + stateErrors.text.slice(-6000);
                return;
            }
            try {
                const result = JSON.parse(stateOutput.text);
                setup.stage = result.stage;
                setup.message = result.message;
            } catch (error) { setup.errorMessage = "Invalid setup status: " + error; }
        }
        // qmllint enable signal-handler-parameters
    }
    Timer { interval: 1000; repeat: true; running: window.visible; onTriggered: setup.refresh() }
    PanelWindow {
        id: window
        visible: false
        implicitWidth: Math.min(Style.space(560), screen ? screen.width - 32 : 560)
        implicitHeight: Math.min(Style.space(560), screen ? screen.height - 32 : 560)
        color: "transparent"
        exclusionMode: ExclusionMode.Ignore
        WlrLayershell.namespace: "fractal-search-setup"
        WlrLayershell.layer: WlrLayer.Overlay
        WlrLayershell.keyboardFocus: WlrKeyboardFocus.OnDemand
        OmarchyUi.BorderSurface {
            anchors.fill: parent
            focus: true
            Keys.onEscapePressed: setup.close()
            color: setup.colors.background
            borderSpec: Border.controlSpec("focus", setup.colors.foreground, setup.colors.accent)
            ScrollView {
                id: scroll
                anchors.fill: parent
                anchors.margins: Style.space(24)
                contentWidth: availableWidth
                clip: true
                ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
                ColumnLayout {
                    width: scroll.availableWidth
                    spacing: Style.space(14)
                    SearchText { text: "Welcome to Fractal Search"; font.bold: true; color: setup.colors.foreground; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    SearchText {
                        Layout.fillWidth: true; wrapMode: Text.WordWrap; color: setup.colors.foreground
                        text: "Find images, text, audio and video by meaning. First, install the search backend. No compiler is needed."
                    }
                    SearchText {
                        Layout.fillWidth: true; wrapMode: Text.WordWrap; color: setup.colors.foreground
                        text: "CPU works without NVIDIA libraries. CUDA requires an NVIDIA GPU, CUDA 12 and cuDNN 9 already installed. Missing dependencies are reported; setup does not install system packages."
                    }
                    SearchText {
                        Layout.fillWidth: true; wrapMode: Text.WordWrap; color: setup.colors.foreground
                        text: "Setup downloads a verified release from GitHub and installs a user service. Your files stay in place. Afterward, select Set up model (about 1.50 GB), then add folders."
                    }
                    SearchText {
                        Layout.fillWidth: true; wrapMode: Text.WordWrap; color: setup.colors.foreground
                        text: "CUDA library folder (optional, for privately installed libraries)"
                    }
                    TextField {
                        id: cudaDirectory
                        objectName: "cudaLibraryDirectory"
                        Layout.fillWidth: true
                        enabled: !setup.busy
                        placeholderText: "/absolute/path/to/lib"
                        color: setup.colors.foreground
                        placeholderTextColor: setup.colors.foreground
                        selectionColor: setup.colors.accent
                        font.family: SearchTheme.font.family
                        font.pixelSize: SearchTheme.font.body
                        background: Rectangle {
                            color: setup.colors.background
                            border.color: cudaDirectory.activeFocus ? setup.colors.accent : setup.colors.foreground
                            border.width: 1
                        }
                    }
                    Flow {
                        Layout.fillWidth: true; spacing: Style.space(8)
                        SearchButton { objectName: "installCpu"; text: "Install CPU"; colors: setup.colors; enabled: !setup.busy; onClicked: setup.install("cpu") }
                        SearchButton { objectName: "installCuda"; text: "Install NVIDIA CUDA"; colors: setup.colors; enabled: !setup.busy; onClicked: setup.install("cuda") }
                    }
                    ProgressBar { Layout.fillWidth: true; visible: setup.busy; indeterminate: true }
                    SearchText { Layout.fillWidth: true; wrapMode: Text.WrapAnywhere; color: setup.stage === "error" ? setup.colors.error : setup.colors.foreground; text: setup.message }
                    SearchText { Layout.fillWidth: true; wrapMode: Text.WrapAnywhere; visible: setup.errorMessage.length > 0; color: setup.colors.error; text: setup.errorMessage }
                    SearchText { Layout.fillWidth: true; wrapMode: Text.WordWrap; visible: setup.busy; color: setup.colors.foreground; text: "You can close this panel; installation continues. Open Fractal Search again when it finishes." }
                    SearchButton { text: "Close"; colors: setup.colors; onClicked: setup.close() }
                }
            }
        }
    }
}
