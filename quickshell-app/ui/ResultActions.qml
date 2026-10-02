import QtQuick
import QtQuick.Layouts
import qs.Commons

RowLayout {
    id: actions
    required property Controller app
    spacing: Style.space(8)
    SearchButton {
        objectName: "openImageButton"
        Layout.fillWidth: true
        text: "Open image"
        Accessible.description: "Open in your image viewer. Keyboard shortcut: Enter."
        colors: actions.app.colors
        filled: true
        enabled: actions.app.ready && !actions.app.loading && !actions.app.closing && !!actions.app.selectedAsset
        onClicked: actions.app.openOriginal()
    }
    SearchButton {
        objectName: "revealImageButton"
        Layout.fillWidth: true
        text: "Show in folder"
        Accessible.description: "Select the original in your file manager. Keyboard shortcut: Ctrl+Enter."
        colors: actions.app.colors
        enabled: actions.app.ready && !actions.app.loading && !actions.app.closing && !!actions.app.selectedAsset
        onClicked: actions.app.revealOriginal()
    }
}
