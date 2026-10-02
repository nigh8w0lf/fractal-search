import QtQuick
import qs.Commons
import QtQuick.Controls

Button {
    id: button
    required property var colors
    property bool primary: false
    property bool filled: false
    property bool selected: false
    property bool quiet: false
    implicitHeight: Style.space(32)
    padding: Style.space(9)
    font.family: SearchTheme.font.family
    font.pixelSize: SearchTheme.font.body
    focusPolicy: Qt.StrongFocus
    contentItem: Text {
        text: button.text
        textFormat: Text.PlainText
        color: button.filled ? button.colors.background : button.primary ? button.colors.accent : button.colors.foreground
        font: button.font
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        opacity: button.enabled ? 1 : 0.4
    }
    background: Rectangle {
        color: button.filled ? (button.down ? Qt.darker(button.colors.accent, 1.15) : button.hovered ? Qt.lighter(button.colors.accent, 1.1) : button.colors.accent) : button.selected || button.hovered || button.down ? Qt.alpha(button.colors.accent, 0.14) : button.quiet ? "transparent" : Qt.alpha(button.colors.foreground, 0.035)
        border.width: button.quiet && !button.activeFocus ? 0 : 1
        border.color: button.activeFocus && button.filled ? button.colors.foreground : button.activeFocus || button.selected ? button.colors.accent : Qt.alpha(button.colors.foreground, 0.3)
        opacity: button.enabled ? 1 : 0.5
    }
}
