import QtQuick
import qs.Commons
import qs.Ui as OmarchyUi
import QtQuick.Controls

Button {
    id: button
    required property var colors
    property bool primary: false
    property bool filled: false
    property bool selected: false
    property bool quiet: false
    implicitHeight: Math.max(Style.space(32), implicitContentHeight + topPadding + bottomPadding)
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
    background: OmarchyUi.BorderSurface {
        color: button.filled ? (button.down ? Qt.darker(button.colors.accent, 1.15) : button.hovered ? Qt.lighter(button.colors.accent, 1.1) : button.colors.accent)
            : button.down ? Style.pressedFillFor(button.colors.foreground, button.colors.accent)
            : button.activeFocus ? Style.focusFillFor(button.colors.foreground, button.colors.accent)
            : button.hovered ? Style.hoverFillFor(button.colors.foreground, button.colors.accent)
            : button.selected ? Style.selectedFillFor(button.colors.foreground, button.colors.accent)
            : button.quiet ? "transparent" : Style.normalFillFor(button.colors.foreground, button.colors.accent)
        borderSpec: button.activeFocus ? Border.controlSpec("focus", button.colors.foreground, button.colors.accent)
            : button.hovered ? Border.controlSpec("hover-cursor", button.colors.foreground, button.colors.accent)
            : button.selected ? Border.controlSpec("selected", button.colors.foreground, button.colors.accent)
            : button.quiet ? Border.none() : Border.controlSpec("normal", button.colors.foreground, button.colors.accent)
        opacity: button.enabled ? 1 : 0.5
    }
}
