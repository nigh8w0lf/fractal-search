import QtQuick
import qs.Commons
import QtQuick.Controls

CheckBox {
    id: control
    required property var colors
    implicitHeight: Style.space(34)
    font.family: SearchTheme.font.family
    font.pixelSize: SearchTheme.font.body
    leftPadding: 0
    rightPadding: Style.space(30)
    contentItem: Text {
        text: control.text
        font: control.font
        color: control.colors.foreground
        verticalAlignment: Text.AlignVCenter
        opacity: control.enabled ? 1 : 0.45
    }
    indicator: Rectangle {
        x: control.width - width
        anchors.verticalCenter: parent.verticalCenter
        width: 19
        height: 19
        color: control.checked ? control.colors.accent : "transparent"
        border.color: control.activeFocus ? control.colors.accent : Qt.alpha(control.colors.foreground, 0.4)
        opacity: control.enabled ? 1 : 0.45
        Text {
            anchors.centerIn: parent
            text: control.checked ? "✓" : ""
            color: control.colors.background
            font.pixelSize: 15
        }
    }
}
