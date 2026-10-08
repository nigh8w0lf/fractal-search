import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import qs.Commons

ColumnLayout {
    id: policy
    required property Controller app
    required property string label
    required property string settingKey
    required property int defaultSeconds
    readonly property int seconds: app.settings[settingKey]
    spacing: Style.space(6)

    SearchText {
        Layout.fillWidth: true
        text: policy.label
        color: policy.app.colors.foreground
        wrapMode: Text.WordWrap
    }
    RowLayout {
        Layout.fillWidth: true
        SearchChoice {
            objectName: policy.settingKey + "Policy"
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            colors: policy.app.colors
            model: ["Release when idle", "Keep warm"]
            currentIndex: policy.seconds === 0 ? 1 : 0
            enabled: policy.app.ready && !policy.app.busy
            onActivated: {
                const change = {};
                change[policy.settingKey] = currentIndex === 1 ? 0 : (policy.seconds > 0 ? policy.seconds : policy.defaultSeconds);
                policy.app.updateSettings(change);
            }
        }
        SpinBox {
            objectName: policy.settingKey + "Duration"
            visible: policy.seconds > 0
            from: 1
            to: 86400
            stepSize: 60
            value: policy.seconds
            editable: true
            enabled: policy.app.ready && !policy.app.busy
            implicitWidth: Style.space(132)
            implicitHeight: Style.space(32)
            font.family: policy.app.monoFont
            font.pixelSize: SearchTheme.font.body
            palette.text: policy.app.colors.foreground
            palette.buttonText: policy.app.colors.foreground
            palette.button: policy.app.colors.surface
            palette.base: policy.app.colors.background
            palette.highlight: policy.app.colors.accent
            Accessible.name: policy.label + " idle timeout in seconds"
            onValueModified: {
                const change = {};
                change[policy.settingKey] = value;
                policy.app.updateSettings(change);
            }
            background: Rectangle {
                color: Qt.alpha(policy.app.colors.foreground, 0.035)
                border.color: policy.app.borderColor
            }
        }
    }
    SearchText {
        Layout.fillWidth: true
        text: policy.seconds === 0 ? "Loaded sessions stay warm until the service stops. Models still load only when needed." : "Release after " + policy.seconds + " seconds without use (1–86,400 seconds). The next request reloads the model."
        color: policy.app.secondaryText
        wrapMode: Text.WordWrap
        font.pixelSize: SearchTheme.font.bodySmall
    }
}
