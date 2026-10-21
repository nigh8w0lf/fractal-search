import QtQuick
import QtQuick.Layouts
import qs.Commons

ColumnLayout {
    id: section
    required property var app
    required property string serverPath
    readonly property bool configured: serverPath.startsWith("/")
    readonly property string configuration: JSON.stringify({mcpServers: {"fractal-search": {command: serverPath, args: []}}}, null, 2)
    property string copied: ""
    signal openGuide
    spacing: Style.space(10)

    function copyValue(value, label) {
        if (!configured) {
            app.fail("The MCP launcher path is missing. Reinstall the complete Fractal Search bundle.");
            return;
        }
        clipboardText.text = value;
        clipboardText.selectAll();
        clipboardText.copy();
        clipboardText.deselect();
        copied = label;
        copiedTimer.restart();
    }
    SearchText {
        text: "Connect an AI agent"
        font.bold: true
        color: section.app.colors.foreground
    }
    SearchText {
        Layout.fillWidth: true
        text: "Let an MCP-compatible AI client search your indexed files. Your client starts the read-only server when needed; there is no separate switch to enable."
        wrapMode: Text.WordWrap
        color: section.app.secondaryText
        font.pixelSize: SearchTheme.font.bodySmall
    }
    SearchText {
        Layout.fillWidth: true
        text: "Agents can read filenames, cached text excerpts and thumbnails, but cannot open originals or change your index. Your AI client controls how it uses this content."
        wrapMode: Text.WordWrap
        color: section.app.secondaryText
        font.pixelSize: SearchTheme.font.bodySmall
    }
    SearchText {
        Layout.fillWidth: true
        text: section.configured ? section.serverPath : "MCP launcher path unavailable. Reinstall the complete app bundle."
        wrapMode: Text.WrapAnywhere
        color: section.configured ? section.app.secondaryText : section.app.colors.error
        font.pixelSize: SearchTheme.font.bodySmall
    }
    Flow {
        Layout.fillWidth: true
        spacing: Style.space(8)
        SearchButton {
            objectName: "copyAgentConfiguration"
            text: section.copied === "configuration" ? "Copied configuration" : "Copy configuration"
            colors: section.app.colors
            enabled: section.configured
            onClicked: section.copyValue(section.configuration, "configuration")
        }
        SearchButton {
            objectName: "copyAgentPath"
            text: section.copied === "path" ? "Copied server path" : "Copy server path"
            colors: section.app.colors
            enabled: section.configured
            onClicked: section.copyValue(section.serverPath, "path")
        }
        SearchButton {
            objectName: "agentSetupGuide"
            text: "Setup guide"
            colors: section.app.colors
            onClicked: section.openGuide()
        }
    }
    TextEdit { id: clipboardText; visible: false }
    Timer { id: copiedTimer; interval: 2500; onTriggered: section.copied = "" }
}
