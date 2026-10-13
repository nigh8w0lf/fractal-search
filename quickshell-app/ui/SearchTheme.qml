pragma Singleton
import QtQuick
import qs.Commons

QtObject {
    // Omarchy exposes this nested object as QObject in QML metadata. Keep its
    // runtime tokens dynamic rather than duplicating the host's font settings.
    readonly property var font: Style.font
    readonly property var popups: Color.popups
}
