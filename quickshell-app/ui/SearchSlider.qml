import QtQuick
import QtQuick.Controls
import qs.Commons

Slider {
    id: control
    required property var colors
    implicitHeight: Style.space(28)
    implicitWidth: Style.space(120)
    focusPolicy: Qt.StrongFocus
    background: Rectangle {
        x: control.leftPadding
        y: control.topPadding + control.availableHeight / 2 - height / 2
        width: control.availableWidth
        height: Style.space(3)
        color: Qt.alpha(control.colors.foreground, 0.25)
        Rectangle {
            width: control.visualPosition * parent.width
            height: parent.height
            color: control.colors.accent
        }
    }
    handle: Rectangle {
        x: control.leftPadding + control.visualPosition * (control.availableWidth - width)
        y: control.topPadding + control.availableHeight / 2 - height / 2
        width: Style.space(12)
        height: width
        radius: width / 2
        color: control.enabled ? control.colors.accent : control.colors.muted
        border.width: control.activeFocus ? 2 : 0
        border.color: control.colors.foreground
    }
}
