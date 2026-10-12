pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import qs.Commons

ColumnLayout {
    id: section
    required property Controller app
    property int maximumExcerptHeight: Style.space(160)
    property bool expanded: false
    readonly property string assetId: app.selectedAsset ? app.selectedAsset.assetId : ""
    readonly property var matches: app.additionalMatches(app.selectedAsset)
    visible: matches.length > 0
    spacing: Style.space(6)
    onAssetIdChanged: expanded = false

    SearchButton {
        objectName: "groupedTextMatchesToggle"
        Layout.fillWidth: true
        colors: section.app.colors
        text: (section.expanded ? "Hide " : "Show ") + section.matches.length + " more matching passage" + (section.matches.length === 1 ? "" : "s")
        Accessible.description: "Additional matching excerpts from the selected file. Open file and Show in folder apply to the original file."
        onClicked: section.expanded = !section.expanded
    }
    ScrollView {
        id: excerptScroll
        visible: section.expanded
        Layout.fillWidth: true
        Layout.preferredHeight: Math.min(section.maximumExcerptHeight, excerpts.implicitHeight)
        contentWidth: availableWidth
        rightPadding: ScrollBar.vertical.implicitWidth + Style.space(4)
        clip: true
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
        ColumnLayout {
            id: excerpts
            width: excerptScroll.availableWidth
            spacing: Style.space(12)
            Repeater {
                model: section.matches
                ColumnLayout {
                    id: passage
                    required property var modelData
                    Layout.fillWidth: true
                    spacing: Style.space(3)
                    SearchText {
                        Layout.fillWidth: true
                        text: section.app.mediaLabel(passage.modelData)
                        color: section.app.colors.foreground
                        wrapMode: Text.WordWrap
                        font.pixelSize: SearchTheme.font.bodySmall
                    }
                    SearchText {
                        Layout.fillWidth: true
                        text: section.app.scoreLabel(passage.modelData)
                        color: section.app.secondaryText
                        visible: text.length > 0
                        font.pixelSize: SearchTheme.font.bodySmall
                    }
                    TextArea {
                        Layout.fillWidth: true
                        text: passage.modelData.textExcerpt
                        textFormat: TextEdit.PlainText
                        readOnly: true
                        selectByMouse: true
                        wrapMode: TextEdit.Wrap
                        color: section.app.secondaryText
                        font.family: SearchTheme.font.family
                        font.pixelSize: SearchTheme.font.bodySmall
                        padding: 0
                        background: null
                    }
                }
            }
        }
    }
}
