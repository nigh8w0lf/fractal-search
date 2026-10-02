import QtQuick
import qs.Commons
import QtQuick.Controls

ComboBox {
    id: control
    required property var colors
    implicitHeight: Style.space(32)
    font.family: SearchTheme.font.family
    font.pixelSize: SearchTheme.font.body
    leftPadding: Style.space(9)
    rightPadding: Style.space(25)
    palette.text: colors.foreground
    palette.buttonText: colors.foreground
    palette.base: colors.background
    palette.window: colors.background
    palette.highlight: colors.surface
    palette.highlightedText: colors.accent
    contentItem: Text {
        text: control.displayText
        font: control.font
        color: control.colors.foreground
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    background: Rectangle {
        color: Qt.alpha(control.colors.foreground, 0.035)
        border.color: control.activeFocus ? control.colors.accent : Qt.alpha(control.colors.foreground, 0.3)
    }
    indicator: Text {
        x: control.width - width - 9
        anchors.verticalCenter: parent.verticalCenter
        text: "⌄"
        color: control.colors.foreground
    }
}
