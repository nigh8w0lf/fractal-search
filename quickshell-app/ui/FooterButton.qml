import QtQuick
import qs.Commons
import QtQuick.Layouts

SearchButton {
    id: button
    required property string shortcut
    implicitHeight: Style.space(26)
    leftPadding: Style.space(7)
    rightPadding: Style.space(7)
    topPadding: Style.space(3)
    bottomPadding: Style.space(3)
    font.pixelSize: SearchTheme.font.bodySmall
    Accessible.name: text
    Accessible.description: "Keyboard shortcut: " + shortcut

    contentItem: RowLayout {
        spacing: Style.space(7)
        opacity: button.enabled ? 1 : 0.4
        ShortcutKey {
            text: button.shortcut
            colors: button.colors
        }
        SearchText {
            text: button.text
            color: button.colors.foreground
            font: button.font
        }
    }
    background: Rectangle {
        color: button.down ? Qt.alpha(button.colors.accent, 0.18) : button.hovered ? Qt.alpha(button.colors.accent, 0.1) : Qt.alpha(button.colors.foreground, 0.025)
        border.width: 1
        border.color: button.activeFocus || button.hovered ? button.colors.accent : Qt.alpha(button.colors.foreground, 0.16)
        opacity: button.enabled ? 1 : 0.4
    }
    HoverHandler {
        enabled: button.enabled
        cursorShape: Qt.PointingHandCursor
    }
}
