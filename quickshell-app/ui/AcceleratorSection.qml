import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import qs.Commons

ColumnLayout {
    id: section
    required property var app
    property bool expanded: false
    property bool copied: false
    property bool checkRequested: false
    readonly property var diagnostics: app.runtimeDiagnostics
    readonly property bool hasReport: !diagnostics.checking && !diagnostics.errorMessage.length && diagnostics.report !== null
    readonly property bool gpuAvailable: hasReport && diagnostics.report.checksPassed === true
    readonly property bool modelReady: app.ready && app.status.textModelLoaded === true
    readonly property bool gpuActive: modelReady && app.status.provider.toLowerCase() === "cuda"
    readonly property bool cpuActive: modelReady && app.status.provider.toLowerCase() === "cpu"
    readonly property string gpuState: gpuActive ? "Active" : diagnostics.checking ? "Checking…" : diagnostics.errorMessage.length ? "Check failed" : gpuAvailable ? "Available" : hasReport ? "Unavailable" : "Not checked"
    readonly property string diagnosticText: "Fractal Search accelerator report\nConfigured provider: " + app.configuredProvider.toUpperCase() + "\nActive accelerator: " + (gpuActive ? "GPU" : cpuActive ? "CPU" : "Not running") + "\n\n" + (diagnostics.errorMessage.length ? diagnostics.errorMessage + "\n\n" : "") + diagnostics.details
    spacing: Style.space(10)

    function ensureChecked() {
        if (!checkRequested && !diagnostics.report && !diagnostics.checking) {
            checkRequested = true;
            diagnostics.refresh();
        }
    }
    onVisibleChanged: if (visible) ensureChecked()
    Component.onCompleted: if (visible) ensureChecked()

    SearchText {
        text: "Accelerator"
        font.bold: true
        color: section.app.colors.foreground
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: Style.space(16)
        SearchText {
            text: "CPU" + (section.cpuActive ? " · Active" : section.app.configuredProvider === "cpu" ? " · Selected" : "")
            color: section.cpuActive ? section.app.colors.accent : section.app.colors.foreground
        }
        SearchText {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            text: "GPU · " + section.gpuState
            color: section.gpuActive || section.gpuAvailable ? section.app.colors.accent : section.app.secondaryText
            opacity: section.gpuActive || section.gpuAvailable ? 1 : 0.65
            wrapMode: Text.WordWrap
        }
        SearchButton {
            objectName: "acceleratorDetails"
            text: section.expanded ? "Details  ▴" : "Details  ▾"
            colors: section.app.colors
            quiet: true
            onClicked: section.expanded = !section.expanded
        }
    }
    SearchText {
        Layout.fillWidth: true
        visible: !section.gpuActive
        text: section.diagnostics.checking ? "Checking GPU requirements…" : section.diagnostics.errorMessage.length ? "Could not check GPU requirements. Open Details to diagnose." : section.gpuAvailable ? (section.app.configuredProvider === "cuda" ? "GPU requirements met. Model startup will verify GPU execution." : "GPU requirements met. Enable CUDA in the service configuration to use it.") : section.hasReport ? "Share the diagnostics with your agent to get GPU acceleration working." : "GPU requirements have not been checked yet."
        color: section.diagnostics.errorMessage.length ? section.app.colors.error : section.app.secondaryText
        font.pixelSize: SearchTheme.font.bodySmall
        wrapMode: Text.WordWrap
    }
    SearchText {
        Layout.fillWidth: true
        visible: section.app.configuredProvider === "cuda" && !section.gpuActive && !section.diagnostics.checking
        text: "CUDA is selected but is not running. Fractal does not switch to CPU automatically."
        color: section.app.secondaryText
        font.pixelSize: SearchTheme.font.bodySmall
        wrapMode: Text.WordWrap
    }
    ColumnLayout {
        visible: section.expanded
        Layout.fillWidth: true
        spacing: Style.space(10)
        SearchText {
            Layout.fillWidth: true
            text: section.diagnostics.summary
            color: section.diagnostics.errorMessage.length ? section.app.colors.error : section.app.secondaryText
            font.pixelSize: SearchTheme.font.bodySmall
            wrapMode: Text.WordWrap
        }
        RowLayout {
            Layout.fillWidth: true
            SearchButton {
                objectName: "copyAcceleratorDiagnostics"
                text: section.copied ? "Copied" : "Copy diagnostics"
                colors: section.app.colors
                enabled: !section.diagnostics.checking && (section.hasReport || section.diagnostics.errorMessage.length > 0)
                onClicked: {
                    diagnosticDetails.selectAll();
                    diagnosticDetails.copy();
                    diagnosticDetails.deselect();
                    diagnosticDetails.cursorPosition = 0;
                    diagnosticScroll.contentItem.contentY = 0;
                    section.copied = true;
                    copiedTimer.restart();
                }
            }
            Item { Layout.fillWidth: true }
            SearchButton {
                text: section.diagnostics.checking ? "Checking…" : "Refresh"
                colors: section.app.colors
                enabled: !section.diagnostics.checking
                onClicked: {
                    section.copied = false;
                    section.diagnostics.refresh();
                }
            }
        }
        ScrollView {
            id: diagnosticScroll
            Layout.fillWidth: true
            Layout.preferredHeight: Style.space(220)
            contentWidth: availableWidth
            clip: true
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
            TextArea {
                id: diagnosticDetails
                text: section.diagnosticText
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.Wrap
                font.family: SearchTheme.font.family
                font.pixelSize: SearchTheme.font.bodySmall
                color: section.app.colors.foreground
                background: Rectangle {
                    color: Qt.alpha(section.app.colors.foreground, 0.035)
                    border.color: section.app.borderColor
                }
            }
        }
    }
    Timer {
        id: copiedTimer
        interval: 2000
        onTriggered: section.copied = false
    }
}
