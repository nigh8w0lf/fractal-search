pragma ComponentBehavior: Bound
import QtQuick
import qs.Commons
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: page
    required property Controller app
    property int currentTab: 0
    signal addFolder
    signal chooseModel
    signal showSkippedFiles
    signal stopService
    signal restartService
    signal removeFolder(int sourceId, string directory)
    spacing: Style.space(20)

    function focusTab(index) {
        selectTab(index);
        tabs.itemAt(index).forceActiveFocus();
    }
    function selectTab(index) {
        if (index < 0 || index > 2)
            throw new Error("Unknown settings tab: " + index);
        currentTab = index;
        settingsScroll.contentItem.contentY = 0;
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: Style.space(4)
        Repeater {
            id: tabs
            model: ["Folders", "Preferences", "Performance"]
            SearchButton {
                required property int index
                required property string modelData
                Layout.fillWidth: true
                Layout.preferredWidth: 1
                text: modelData
                colors: page.app.colors
                primary: page.currentTab === index
                selected: page.currentTab === index
                Accessible.role: Accessible.PageTab
                Accessible.selected: page.currentTab === index
                onClicked: page.selectTab(index)
                Keys.onLeftPressed: page.focusTab((index + 2) % 3)
                Keys.onRightPressed: page.focusTab((index + 1) % 3)
            }
        }
    }
    ScrollView {
        id: settingsScroll
        Layout.fillWidth: true
        Layout.fillHeight: true
        contentWidth: availableWidth
        clip: true
        rightPadding: ScrollBar.vertical.implicitWidth + 8
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
        ColumnLayout {
            width: settingsScroll.availableWidth
            spacing: Style.space(20)
            ColumnLayout {
                visible: page.currentTab === 0
                Layout.fillWidth: true
                spacing: Style.space(20)
                ColumnLayout {
                    visible: !page.app.status.modelsPresent
                    Layout.fillWidth: true
                    SearchText {
                        Layout.fillWidth: true
                        text: "Set up the search engine before adding your first folder."
                        color: page.app.secondaryText
                        wrapMode: Text.WordWrap
                    }
                    SearchButton {
                        text: "Set up search"
                        colors: page.app.colors
                        onClicked: page.selectTab(2)
                    }
                }
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: Style.space(6)
                    SearchText {
                        text: "Search folders"
                        font.bold: true
                        color: page.app.colors.foreground
                        Layout.bottomMargin: Style.space(3)
                    }
                    Repeater {
                        model: page.app.status.catalog.sources
                        Rectangle {
                            id: folder
                            required property var modelData
                            Layout.fillWidth: true
                            implicitHeight: Style.space(66)
                            color: Qt.alpha(page.app.colors.foreground, 0.035)
                            border.color: page.app.borderColor
                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: Style.space(10)
                                spacing: Style.space(4)
                                ColumnLayout {
                                    Layout.fillWidth: true
                                    Layout.minimumWidth: 0
                                    spacing: Style.space(3)
                                    SearchText {
                                        Layout.fillWidth: true
                                        text: folder.modelData.directory.split("/").filter(Boolean).pop() || folder.modelData.directory
                                        color: page.app.colors.foreground
                                        font.bold: true
                                        elide: Text.ElideMiddle
                                    }
                                    SearchText {
                                        Layout.fillWidth: true
                                        text: page.app.displayPath(folder.modelData.directory)
                                        color: page.app.secondaryText
                                        font.pixelSize: SearchTheme.font.bodySmall
                                        elide: Text.ElideMiddle
                                    }
                                }
                                SearchText {
                                    text: folder.modelData.imageCount + " photos"
                                    color: page.app.secondaryText
                                    font.pixelSize: SearchTheme.font.bodySmall
                                }
                                SearchButton {
                                    text: "×"
                                    Accessible.name: "Remove folder from index"
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
                        text: "Add a folder to make its photos searchable. Your originals stay in place."
                        wrapMode: Text.WordWrap
                        color: page.app.secondaryText
                        Layout.fillWidth: true
                        Layout.bottomMargin: Style.space(4)
                    }
                    SearchButton {
                        Layout.fillWidth: true
                        text: "+  Add folder"
                        filled: true
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
                        text: "Keep folders up to date"
                        colors: page.app.colors
                        checked: page.app.settings.watchChanges
                        enabled: page.app.ready && !page.app.busy
                        onClicked: page.app.updateSettings({
                            watchChanges: checked
                        })
                    }
                    SearchText {
                        Layout.fillWidth: true
                        Layout.bottomMargin: Style.space(8)
                        text: "Automatically index new and changed photos."
                        color: page.app.secondaryText
                        font.pixelSize: SearchTheme.font.bodySmall
                        wrapMode: Text.WordWrap
                    }
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
                        text: "Rescan folders"
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
            }
            ColumnLayout {
                visible: page.currentTab === 1
                Layout.fillWidth: true
                spacing: Style.space(20)
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 0
                    SearchToggle {
                        Layout.fillWidth: true
                        text: "Prepare search at login"
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
                        text: "Load the search model in the background for faster searches."
                        color: page.app.secondaryText
                        font.pixelSize: SearchTheme.font.bodySmall
                        wrapMode: Text.WordWrap
                    }
                    SearchToggle {
                        Layout.fillWidth: true
                        text: "Hide after opening a photo"
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
                SearchText {
                    text: "Search results"
                    color: page.app.colors.foreground
                    font.bold: true
                }
                RowLayout {
                    Layout.fillWidth: true
                    SearchText {
                        text: "Maximum results"
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
            }
            ColumnLayout {
                visible: page.currentTab === 2
                Layout.fillWidth: true
                spacing: Style.space(20)
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: Style.space(6)
                    RowLayout {
                        Layout.fillWidth: true
                        SearchText {
                            text: "Search engine"
                            font.bold: true
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
                AcceleratorSection {
                    Layout.fillWidth: true
                    app: page.app
                }
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: Style.space(6)
                    RowLayout {
                        Layout.fillWidth: true
                        SearchText {
                            text: "Image batch size"
                            color: page.app.colors.foreground
                            Layout.fillWidth: true
                        }
                        SearchChoice {
                            objectName: "imageBatchSizeChoice"
                            colors: page.app.colors
                            model: ["1", "2", "3", "4", "5", "6", "7", "8"]
                            implicitWidth: Style.space(90)
                            currentIndex: page.app.settings.imageBatchSize - 1
                            enabled: page.app.ready && !page.app.busy
                            onActivated: page.app.updateSettings({imageBatchSize: currentIndex + 1})
                        }
                    }
                    SearchText {
                        Layout.fillWidth: true
                        text: "Maximum images processed together during indexing. Larger batches use more memory. Restart Fractal Search to apply changes."
                        color: page.app.secondaryText
                        wrapMode: Text.WordWrap
                        font.pixelSize: SearchTheme.font.bodySmall
                    }
                    RowLayout {
                        visible: page.app.batchRestartRequired
                        Layout.fillWidth: true
                        SearchText {
                            Layout.fillWidth: true
                            text: "Restart required · currently " + page.app.status.activeImageBatchSize + " image(s) per batch"
                            color: page.app.secondaryText
                            wrapMode: Text.WordWrap
                            font.pixelSize: SearchTheme.font.bodySmall
                        }
                        SearchButton {
                            text: "Restart"
                            colors: page.app.colors
                            enabled: page.app.ready && !page.app.busy
                            onClicked: page.restartService()
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
    }
}
