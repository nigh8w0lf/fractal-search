pragma ComponentBehavior: Bound
import QtQuick
import qs.Commons
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: page
    required property Controller app
    signal addFolder
    signal chooseModel
    signal showSkippedFiles
    signal stopService
    signal removeFolder(int sourceId, string directory)
    contentWidth: availableWidth
    clip: true
    // Reserve the gutter even when the thumb fades out, so controls and text
    // never sit under the scrollbar or change width when scrolling begins.
    rightPadding: ScrollBar.vertical.implicitWidth + 12
    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

    ColumnLayout {
        width: page.availableWidth
        spacing: Style.space(16)
        RowLayout {
            Layout.fillWidth: true
            SearchText {
                text: "Top matches"
                color: page.app.colors.foreground
                Layout.fillWidth: true
            }
            SpinBox {
                id: topMatches
                from: 1
                to: 100
                value: page.app.settings.topMatches
                editable: true
                enabled: page.app.ready && !page.app.busy
                implicitWidth: Style.space(130)
                implicitHeight: Style.space(32)
                font.family: page.app.monoFont
                font.pixelSize: SearchTheme.font.body
                palette.text: page.app.colors.foreground
                palette.buttonText: page.app.colors.foreground
                palette.button: page.app.colors.surface
                palette.base: page.app.colors.background
                palette.highlight: page.app.colors.accent
                onValueModified: page.app.updateSettings({
                    topMatches: value
                })
                background: Rectangle {
                    color: Qt.alpha(page.app.colors.foreground, 0.035)
                    border.color: page.app.borderColor
                }
            }
        }
        RowLayout {
            Layout.fillWidth: true
            SearchText {
                text: "Result layout"
                color: page.app.colors.foreground
                Layout.fillWidth: true
            }
            SearchChoice {
                id: resultLayout
                colors: page.app.colors
                model: ["List + preview", "List"]
                // Reserve the longest label even when the shorter one is selected.
                implicitWidth: Math.ceil(layoutLabel.advanceWidth) + leftPadding + rightPadding
                Layout.minimumWidth: implicitWidth
                currentIndex: page.app.settings.resultLayout === "listPreview" ? 0 : 1
                enabled: page.app.ready && !page.app.busy
                onActivated: page.app.updateSettings({
                    resultLayout: currentIndex === 0 ? "listPreview" : "list"
                })
                TextMetrics {
                    id: layoutLabel
                    font: resultLayout.font
                    text: "List + preview"
                }
            }
        }
        ColumnLayout {
            Layout.fillWidth: true
            spacing: Style.space(6)
            SearchText {
                text: "Indexed folders"
                color: page.app.colors.foreground
                Layout.bottomMargin: Style.space(3)
            }
            Repeater {
                model: page.app.status.catalog.sources
                Rectangle {
                    id: folder
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: Style.space(35)
                    color: Qt.alpha(page.app.colors.foreground, 0.035)
                    border.color: page.app.borderColor
                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: Style.space(10)
                        spacing: Style.space(4)
                        SearchText {
                            text: page.app.displayPath(folder.modelData.directory)
                            color: page.app.colors.foreground
                            elide: Text.ElideMiddle
                            Layout.fillWidth: true
                            Layout.minimumWidth: 0
                        }
                        SearchText {
                            text: folder.modelData.imageCount
                            color: page.app.secondaryText
                            font.pixelSize: SearchTheme.font.bodySmall
                        }
                        SearchButton {
                            text: "×"
                            colors: page.app.colors
                            quiet: true
                            enabled: page.app.ready && !page.app.busy
                            onClicked: page.removeFolder(folder.modelData.id, folder.modelData.directory)
                        }
                    }
                }
            }
            SearchText {
                visible: !page.app.status.catalog.sources.length
                text: "No folders indexed yet."
                color: page.app.secondaryText
                Layout.fillWidth: true
                Layout.bottomMargin: Style.space(4)
            }
            SearchButton {
                Layout.fillWidth: true
                text: "+  Add folder"
                colors: page.app.colors
                primary: true
                enabled: page.app.ready && page.app.status.modelsPresent && !page.app.busy
                onClicked: page.addFolder()
            }
        }
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 0
            SearchToggle {
                Layout.fillWidth: true
                text: "Load search model at login"
                colors: page.app.colors
                checked: page.app.startAtLogin
                enabled: page.app.manageService && page.app.loginSettingKnown && !page.app.loginSettingBusy && !page.app.closing
                nextCheckState: function () {
                    return checkState;
                }
                onClicked: page.app.setStartAtLogin(!page.app.startAtLogin)
            }
            SearchText {
                Layout.fillWidth: true
                Layout.bottomMargin: Style.space(8)
                text: "Preload the model in the background at login. Opening Fractal always starts the service."
                color: page.app.secondaryText
                font.pixelSize: SearchTheme.font.bodySmall
                wrapMode: Text.WordWrap
            }
            SearchToggle {
                Layout.fillWidth: true
                text: "Watch for changes"
                colors: page.app.colors
                checked: page.app.settings.watchChanges
                enabled: page.app.ready && !page.app.busy
                onClicked: page.app.updateSettings({
                    watchChanges: checked
                })
            }
            SearchToggle {
                Layout.fillWidth: true
                text: "Hide panel after opening an image"
                colors: page.app.colors
                checked: page.app.settings.closeAfterOpening
                enabled: page.app.ready && !page.app.busy
                onClicked: page.app.updateSettings({
                    closeAfterOpening: checked
                })
            }
        }
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 1
            color: page.app.borderColor
        }
        RowLayout {
            Layout.fillWidth: true
            Rectangle {
                implicitWidth: Style.space(8)
                implicitHeight: Style.space(8)
                radius: 4
                color: page.app.indexing ? page.app.colors.accent : page.app.ready && page.app.status.modelsPresent ? page.app.colors.success : page.app.colors.error
            }
            SearchText {
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                text: (page.app.indexing ? "Indexing" : page.app.ready ? "Index ready" : "Connecting") + " · " + page.app.status.catalog.imageCount.toLocaleString() + " images"
                color: page.app.colors.foreground
                wrapMode: Text.WordWrap
            }
            SearchButton {
                text: "Reindex"
                colors: page.app.colors
                primary: true
                enabled: page.app.ready && page.app.status.modelsPresent && !page.app.busy && page.app.status.catalog.sources.length > 0
                onClicked: page.app.reindexAll()
            }
        }
        RowLayout {
            visible: page.app.indexFailureCount > 0
            Layout.fillWidth: true
            SearchText {
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                text: "Last scan · " + page.app.indexFailureCount + " skipped"
                color: page.app.colors.error
                wrapMode: Text.WordWrap
            }
            SearchButton {
                text: "Skipped files"
                colors: page.app.colors
                onClicked: page.showSkippedFiles()
            }
        }
        ColumnLayout {
            Layout.fillWidth: true
            spacing: Style.space(6)
            RowLayout {
                Layout.fillWidth: true
                SearchText {
                    text: "Search model"
                    color: page.app.colors.foreground
                    Layout.fillWidth: true
                }
                SearchButton {
                    text: "Set up model"
                    colors: page.app.colors
                    enabled: page.app.ready && !page.app.busy
                    onClicked: page.app.installModels()
                }
                SearchButton {
                    text: "Choose folder"
                    colors: page.app.colors
                    enabled: page.app.ready && !page.app.busy
                    onClicked: page.chooseModel()
                }
            }
            SearchText {
                Layout.fillWidth: true
                text: page.app.displayPath(page.app.status.modelDirectory)
                color: page.app.secondaryText
                elide: Text.ElideMiddle
                font.pixelSize: SearchTheme.font.bodySmall
            }
            SearchText {
                text: "FG-CLIP2 · " + page.app.status.provider + " · " + page.app.status.threads + " threads"
                color: page.app.secondaryText
                font.pixelSize: SearchTheme.font.bodySmall
            }
        }
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 1
            color: page.app.borderColor
        }
        ColumnLayout {
            Layout.fillWidth: true
            spacing: Style.space(6)
            RowLayout {
                Layout.fillWidth: true
                SearchText {
                    text: "CUDA accelerator diagnostics"
                    color: page.app.colors.foreground
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                }
                SearchButton {
                    text: page.app.runtimeDiagnostics.checking ? "Checking…" : page.app.runtimeDiagnostics.report ? "Refresh" : "Check"
                    colors: page.app.colors
                    enabled: !page.app.runtimeDiagnostics.checking
                    onClicked: page.app.runtimeDiagnostics.refresh()
                }
            }
            SearchText {
                Layout.fillWidth: true
                text: page.app.runtimeDiagnostics.summary
                color: page.app.runtimeDiagnostics.errorMessage.length ? page.app.colors.error : page.app.secondaryText
                font.pixelSize: SearchTheme.font.bodySmall
                wrapMode: Text.WordWrap
            }
            SearchText {
                Layout.fillWidth: true
                text: "Configured: " + page.app.configuredProvider.toUpperCase() + (page.app.configuredProvider === "cuda" ? " · device " + page.app.configuredDeviceId : "") + ". Checks work even when the search service cannot start."
                color: page.app.secondaryText
                font.pixelSize: SearchTheme.font.bodySmall
                wrapMode: Text.WordWrap
            }
            ColumnLayout {
                visible: page.app.runtimeDiagnostics.details.length > 0
                Layout.fillWidth: true
                spacing: Style.space(6)
                SearchButton {
                    text: "Copy details"
                    colors: page.app.colors
                    onClicked: {
                        diagnosticDetails.selectAll();
                        diagnosticDetails.copy();
                        diagnosticDetails.deselect();
                    }
                }
                ScrollView {
                    Layout.fillWidth: true
                    Layout.preferredHeight: Style.space(240)
                    contentWidth: availableWidth
                    clip: true
                    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
                    TextArea {
                        id: diagnosticDetails
                        text: page.app.runtimeDiagnostics.details
                        readOnly: true
                        selectByMouse: true
                        wrapMode: TextEdit.Wrap
                        font.family: page.app.monoFont
                        font.pixelSize: SearchTheme.font.bodySmall
                        color: page.app.colors.foreground
                        background: Rectangle {
                            color: Qt.alpha(page.app.colors.foreground, 0.035)
                            border.color: page.app.borderColor
                        }
                    }
                }
            }
        }
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 1
            color: page.app.borderColor
        }
        ColumnLayout {
            Layout.fillWidth: true
            spacing: Style.space(6)
            SearchText {
                text: "Shortcut"
                color: page.app.colors.foreground
            }
            SearchText {
                Layout.fillWidth: true
                text: "Bind a desktop shortcut to the command below. Esc goes back or dismisses."
                color: page.app.secondaryText
                wrapMode: Text.WordWrap
                font.pixelSize: SearchTheme.font.bodySmall
            }
            TextArea {
                Layout.fillWidth: true
                text: "omarchy-shell shell toggle fractal.search"
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.Wrap
                font.family: page.app.monoFont
                font.pixelSize: SearchTheme.font.bodySmall
                color: page.app.colors.foreground
                background: Rectangle {
                    color: Qt.alpha(page.app.colors.foreground, 0.035)
                    border.color: page.app.borderColor
                }
            }
        }
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 1
            color: page.app.borderColor
        }
        RowLayout {
            Layout.fillWidth: true
            SearchText {
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                text: "Closing the panel keeps search ready. Stop the service to free model memory; opening Fractal starts it again."
                color: page.app.secondaryText
                wrapMode: Text.WordWrap
                font.pixelSize: SearchTheme.font.bodySmall
            }
            FooterButton {
                text: "Stop service"
                shortcut: "Ctrl+Q"
                colors: page.app.colors
                enabled: !page.app.closing
                onClicked: page.stopService()
            }
        }
    }
}
