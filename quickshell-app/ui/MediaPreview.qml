pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import QtMultimedia
import qs.Commons

ColumnLayout {
    id: root
    required property var app
    property bool active: false
    readonly property var asset: app.selectedAsset
    readonly property int selection: app.previewGeneration
    readonly property bool isVideo: !!asset && asset.mediaKind === "video"
    readonly property bool playing: player.playbackState === MediaPlayer.PlayingState
    readonly property int position: player.position
    readonly property int duration: player.duration
    readonly property bool loaded: player.source.toString().length > 0
    property bool loading: false
    property bool pendingPlay: false
    property string playbackError: ""
    property int requestSerial: 0
    spacing: Style.space(8)

    function reset() {
        ++requestSerial;
        pendingPlay = false;
        loading = false;
        timeout.stop();
        player.stop();
        player.source = "";
        playbackError = "";
    }
    function fail(message) {
        reset();
        playbackError = message + " You can also use Open file.";
    }
    function startAtMatch() {
        const start = asset && asset.startMs !== null ? Number(asset.startMs) : 0;
        if (start > 0 && (!player.seekable || start >= player.duration)) {
            fail("The matched timestamp cannot be played in this file. Rescan it if it has changed.");
            return;
        }
        if (player.hasAudio && !audio.muted && devices.audioOutputs.length === 0) {
            fail("No audio output device is available. Connect an output device or mute the preview.");
            return;
        }
        player.position = start;
        player.play();
    }
    function togglePlayback() {
        if (!active || !asset || loading)
            return;
        if (playing) {
            player.pause();
            return;
        }
        if (loaded) {
            if (player.hasAudio && !audio.muted && devices.audioOutputs.length === 0) {
                fail("No audio output device is available. Connect an output device or mute the preview.");
                return;
            }
            if (player.mediaStatus === MediaPlayer.EndOfMedia)
                startAtMatch();
            else
                player.play();
            return;
        }
        playbackError = "";
        loading = true;
        timeout.restart();
        const serial = ++requestSerial;
        const id = asset.assetId;
        app.request({type: "originalUrl", assetId: id}, function(result, error) {
            if (serial !== root.requestSerial || !root.active || !root.asset || root.asset.assetId !== id)
                return;
            if (!result || typeof result.url !== "string" || !result.url.startsWith("file:///")) {
                root.fail(error || "The original file is unavailable for local playback.");
                return;
            }
            root.pendingPlay = true;
            player.source = result.url;
        });
    }
    function seek(milliseconds) {
        if (loaded && player.seekable)
            player.position = Math.max(0, Math.min(player.duration, milliseconds));
    }
    function clock(milliseconds) {
        const seconds = Math.floor(Math.max(0, milliseconds) / 1000);
        return Math.floor(seconds / 60) + ":" + (seconds % 60).toString().padStart(2, "0");
    }
    onSelectionChanged: reset()
    onActiveChanged: if (!active) reset()
    Component.onDestruction: player.stop()
    Connections {
        target: root.app
        function onExternalAppOpened(): void { root.reset(); }
    }
    Timer {
        id: timeout
        interval: 15000
        onTriggered: root.fail("Loading the media timed out. Check that the original file is accessible.")
    }
    MediaDevices {
        id: devices
        onAudioOutputsChanged: {
            if (root.playing && player.hasAudio && !audio.muted && audioOutputs.length === 0)
                root.fail("No audio output device is available. Connect an output device or mute the preview.");
        }
    }
    MediaPlayer {
        id: player
        autoPlay: false
        audioOutput: AudioOutput { id: audio; volume: volumeControl.value }
        videoOutput: video
        onErrorOccurred: (error, errorString) => root.fail("Playback failed: " + errorString)
        onMediaStatusChanged: {
            if (root.pendingPlay && mediaStatus === MediaPlayer.LoadedMedia) {
                root.pendingPlay = false;
                root.loading = false;
                timeout.stop();
                if (root.isVideo && !hasVideo) {
                    root.fail("The file has no playable video track.");
                    return;
                }
                if (!root.isVideo && !hasAudio) {
                    root.fail("The file has no playable audio track.");
                    return;
                }
                player.activeVideoTrack = root.isVideo ? 0 : -1;
                root.startAtMatch();
            }
        }
    }
    Item {
        Layout.fillWidth: true
        Layout.fillHeight: true
        Layout.minimumHeight: Style.space(45)
        clip: true
        VideoOutput {
            id: video
            anchors.fill: parent
            visible: root.isVideo && root.loaded
            fillMode: VideoOutput.PreserveAspectFit
        }
        Image {
            anchors.fill: parent
            visible: root.isVideo && !root.loaded
            source: visible ? root.app.previewUrl : ""
            asynchronous: true
            fillMode: Image.PreserveAspectFit
            onStatusChanged: if (status === Image.Error) root.playbackError = "The cached poster could not be displayed. Press Play to load the original video."
        }
        SearchText {
            anchors.centerIn: parent
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
            visible: !root.isVideo || root.loading || root.playbackError.length > 0
            text: root.playbackError || (root.loading ? "Loading media…" : "Audio preview")
            color: root.playbackError ? root.app.colors.error : root.app.secondaryText
            wrapMode: Text.Wrap
        }
    }
    RowLayout {
        Layout.fillWidth: true
        SearchSlider {
            id: timeline
            objectName: "playbackSeek"
            Layout.fillWidth: true
            colors: root.app.colors
            from: 0
            to: Math.max(1, player.duration)
            value: player.position
            enabled: root.loaded && player.seekable && !root.loading
            onMoved: root.seek(value)
            Accessible.name: "Playback position"
        }
        SearchText {
            text: root.clock(player.position) + " / " + root.clock(player.duration)
            color: root.app.secondaryText
            font.pixelSize: SearchTheme.font.bodySmall
        }
    }
    RowLayout {
        Layout.fillWidth: true
        SearchButton {
            objectName: "playbackToggle"
            colors: root.app.colors
            text: root.loading ? "Loading…" : root.playing ? "Pause" : "Play"
            enabled: root.active && root.app.ready && !root.loading
            onClicked: root.togglePlayback()
        }
        SearchButton {
            objectName: "playbackMute"
            colors: root.app.colors
            text: audio.muted ? "Unmute" : "Mute"
            onClicked: {
                if (audio.muted && root.playing && player.hasAudio && devices.audioOutputs.length === 0) {
                    root.fail("No audio output device is available. Connect an output device to unmute.");
                    return;
                }
                audio.muted = !audio.muted;
            }
        }
        SearchSlider {
            id: volumeControl
            objectName: "playbackVolume"
            Layout.fillWidth: true
            Layout.minimumWidth: Style.space(35)
            colors: root.app.colors
            from: 0
            to: 1
            value: 0.8
            Accessible.name: "Volume"
        }
    }
}
