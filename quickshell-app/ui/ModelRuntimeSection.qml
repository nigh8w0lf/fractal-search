pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import qs.Commons

ColumnLayout {
    id: section
    required property Controller app
    property bool expanded: false
    readonly property var diagnostics: app.ready ? app.status.runtimeDiagnostics : null
    spacing: Style.space(10)

    function duration(value) {
        if (value === null || value === undefined)
            return "not recorded";
        return value < 1000 ? Math.round(value) + " ms" : (value / 1000).toFixed(2) + " s";
    }
    function summary(name, model) {
        if (!model)
            return name + ": unavailable";
        return name + ": " + (model.loaded ? "loaded" : "released / not loaded") + " · " + (model.idleTimeoutSeconds === null || model.idleTimeoutSeconds === 0 ? "keep warm" : "release after " + model.idleTimeoutSeconds + " s idle");
    }
    SearchText {
        Layout.fillWidth: true
        text: "Model memory"
        color: section.app.colors.foreground
        font.bold: true
    }
    ModelIdlePolicy {
        Layout.fillWidth: true
        app: section.app
        label: "Text search model"
        settingKey: "textIdleSeconds"
        defaultSeconds: 900
    }
    ModelIdlePolicy {
        Layout.fillWidth: true
        app: section.app
        label: "Vision and audio models"
        settingKey: "mediaIdleSeconds"
        defaultSeconds: 300
    }
    SearchText {
        Layout.fillWidth: true
        text: "Changes apply immediately. Idle release waits for active inference to finish. Session state below does not measure RAM or GPU memory returned by the runtime."
        color: section.app.secondaryText
        wrapMode: Text.WordWrap
        font.pixelSize: SearchTheme.font.bodySmall
    }
    RowLayout {
        Layout.fillWidth: true
        SearchButton {
            Layout.fillWidth: true
            objectName: "modelRuntimeDetailsToggle"
            colors: section.app.colors
            text: section.expanded ? "Hide session details" : "Session details"
            onClicked: {
                section.expanded = !section.expanded;
                if (section.expanded && section.app.ready && !section.app.busy)
                    section.app.refreshStatus();
            }
        }
        SearchButton {
            visible: section.expanded
            text: "Refresh"
            colors: section.app.colors
            enabled: section.app.ready && !section.app.busy
            onClicked: section.app.refreshStatus()
        }
    }
    ColumnLayout {
        visible: section.expanded
        Layout.fillWidth: true
        spacing: Style.space(10)
        Repeater {
            model: [{key: "text", label: "Text"}, {key: "vision", label: "Vision"}, {key: "audio", label: "Audio"}]
            ColumnLayout {
                id: modality
                required property var modelData
                readonly property var details: section.diagnostics ? section.diagnostics[modelData.key] : null
                Layout.fillWidth: true
                spacing: Style.space(3)
                SearchText {
                    Layout.fillWidth: true
                    text: section.summary(modality.modelData.label, modality.details)
                    color: section.app.colors.foreground
                    wrapMode: Text.WordWrap
                    font.pixelSize: SearchTheme.font.bodySmall
                }
                SearchText {
                    Layout.fillWidth: true
                    visible: !!modality.details
                    text: modality.details ? "Loads: " + modality.details.loadCount + " · releases: " + modality.details.releaseCount + " · idle: " + section.duration(modality.details.idleForMs) + "\nLast load: " + section.duration(modality.details.lastLoadDurationMs) + " · last reload: " + section.duration(modality.details.lastReloadDurationMs) : ""
                    color: section.app.secondaryText
                    wrapMode: Text.WordWrap
                    font.pixelSize: SearchTheme.font.bodySmall
                }
            }
        }
        SearchText {
            Layout.fillWidth: true
            text: "Service snapshot, refreshed every 5 seconds while Performance is visible and the service is idle. Refresh to check now."
            color: section.app.secondaryText
            wrapMode: Text.WordWrap
            font.pixelSize: SearchTheme.font.bodySmall
        }
    }
}
