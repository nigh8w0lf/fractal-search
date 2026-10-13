import QtQuick
import qs.Commons
import QtQuick.Layouts

SearchButton {
    id: button
    required property string shortcut
    quiet: true
    implicitHeight: Math.max(Style.space(28), implicitContentHeight + topPadding + bottomPadding)
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
    HoverHandler {
        enabled: button.enabled
        cursorShape: Qt.PointingHandCursor
    }
}
