import QtQuick
import qs.Commons

Rectangle {
    id: keycap
    required property string text
    required property var colors
    implicitWidth: label.implicitWidth + 10
    implicitHeight: Style.space(18)
    color: Qt.alpha(colors.foreground, 0.06)
    border.width: 1
    border.color: Qt.alpha(colors.foreground, 0.18)

    SearchText {
        id: label
        anchors.centerIn: parent
        text: keycap.text
        color: Qt.alpha(keycap.colors.foreground, 0.75)
        font.pixelSize: SearchTheme.font.caption
    }
}
