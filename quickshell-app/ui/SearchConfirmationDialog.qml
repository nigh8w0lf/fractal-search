import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import qs.Commons

Dialog {
    id: dialog
    required property var colors
    required property color borderColor
    required property string message
    required property string confirmText
    property string cancelText: "Cancel"
    readonly property bool cancelFocused: cancelButton.activeFocus
    readonly property Item confirmButton: acceptButton

    anchors.centerIn: parent
    width: Math.min(500, parent.width - 32)
    height: Math.min(implicitHeight, parent.height - 24)
    padding: Style.space(20)
    spacing: Style.space(12)
    modal: true
    popupType: Popup.Item
    focus: true
    closePolicy: Popup.CloseOnEscape
    enter: Transition {}
    exit: Transition {}
    header: null
    background: Rectangle {
        color: dialog.colors.background
        border.color: dialog.borderColor
    }
    contentItem: ColumnLayout {
        spacing: Style.space(12)
        SearchText {
            Layout.fillWidth: true
            text: dialog.title
            color: dialog.colors.foreground
            font.bold: true
            font.pixelSize: SearchTheme.font.title
            wrapMode: Text.WordWrap
        }
        Flickable {
            id: descriptionScroll
            Layout.fillWidth: true
            Layout.fillHeight: true
            implicitHeight: description.implicitHeight
            contentWidth: width
            contentHeight: description.height
            boundsBehavior: Flickable.StopAtBounds
            clip: true
            ScrollBar.vertical: ScrollBar {}
            TextArea {
                id: description
                width: descriptionScroll.width
                height: implicitHeight
                text: dialog.message
                textFormat: TextEdit.PlainText
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.Wrap
                color: dialog.colors.foreground
                font.family: SearchTheme.font.family
                font.pixelSize: SearchTheme.font.body
                padding: 0
                background: null
            }
        }
    }
    footer: Item {
        implicitHeight: actions.implicitHeight + Style.space(20)
        RowLayout {
            id: actions
            anchors.right: parent.right
            anchors.rightMargin: Style.space(20)
            anchors.verticalCenter: parent.verticalCenter
            spacing: Style.space(8)
            SearchButton {
                id: cancelButton
                text: dialog.cancelText
                colors: dialog.colors
                implicitWidth: Math.max(Style.space(88), implicitContentWidth + leftPadding + rightPadding)
                implicitHeight: Style.space(34)
                KeyNavigation.right: acceptButton
                KeyNavigation.tab: acceptButton
                KeyNavigation.backtab: acceptButton
                Keys.onReturnPressed: dialog.reject()
                Keys.onEnterPressed: dialog.reject()
                onClicked: dialog.reject()
            }
            SearchButton {
                id: acceptButton
                text: dialog.confirmText
                colors: dialog.colors
                implicitHeight: Style.space(34)
                KeyNavigation.left: cancelButton
                KeyNavigation.tab: cancelButton
                KeyNavigation.backtab: cancelButton
                Keys.onReturnPressed: dialog.accept()
                Keys.onEnterPressed: dialog.accept()
                onClicked: dialog.accept()
            }
        }
    }
    onOpened: cancelButton.forceActiveFocus()
}
