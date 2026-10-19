#!/usr/bin/env python3
"""Synthetic Qt Multimedia playback regression, with an optional private compositor.

No catalog, models, user media or shared-desktop input. Native mode requires
Hyprland, hyprctl and grim. Artifacts are retained in the printed temp directory.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='fsp-', dir='/tmp'))
    print(f'Playback test artifacts: {root}', flush=True)
    runtime = root / 'run'; runtime.mkdir(mode=0o700)
    env = dict(os.environ, XDG_RUNTIME_DIR=str(runtime), QT_QPA_PLATFORM='offscreen',
               QT_QUICK_BACKEND='software', QT_QUICK_CONTROLS_STYLE='Basic', QT_QPA_PLATFORMTHEME='generic')
    for key in ['DISPLAY', 'WAYLAND_DISPLAY', 'HYPRLAND_INSTANCE_SIGNATURE', 'DBUS_SESSION_BUS_ADDRESS']:
        env.pop(key, None)
    children = []
    logs = []
    def spawn(command, name):
        log = (root / (name + '.log')).open('w'); logs.append(log)
        child = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        children.append(child)
        return child
    try:
        if args.native:
            env['WAYLAND_DISPLAY'] = str(Path(os.environ['XDG_RUNTIME_DIR']) / os.environ['WAYLAND_DISPLAY'])
            config = root / 'hyprland.lua'
            config.write_text('hl.monitor({output="", mode="1280x900@60", position="auto", scale=1})\nhl.config({animations={enabled=false},misc={disable_hyprland_logo=true,disable_splash_rendering=true}})\n')
            compositor = spawn(['Hyprland', '-c', str(config)], 'hyprland')
            deadline = time.monotonic() + 20
            while not list(runtime.glob('wayland-*.lock')):
                assert compositor.poll() is None
                assert time.monotonic() < deadline
                time.sleep(.05)
            env['WAYLAND_DISPLAY'] = list(runtime.glob('wayland-*.lock'))[0].name.removesuffix('.lock')
            env['HYPRLAND_INSTANCE_SIGNATURE'] = next((runtime / 'hypr').iterdir()).name
            def ctl(*command):
                return subprocess.check_output(['hyprctl', *command], env=env, text=True)
            deadline = time.monotonic() + 15
            while True:
                probe = subprocess.run(['hyprctl', '-j', 'monitors'], env=env, capture_output=True, text=True, timeout=3)
                outputs = json.loads(probe.stdout) if probe.returncode == 0 else []
                if any(output.get('width', 0) > 0 for output in outputs):
                    break
                assert compositor.poll() is None
                assert time.monotonic() < deadline, probe.stdout + probe.stderr
                time.sleep(.05)
            ctl('output', 'create', 'headless', 'playback-test')
            for output in outputs:
                ctl('output', 'remove', output['name'])
            deadline = time.monotonic() + 15
            while True:
                outputs = json.loads(ctl('-j', 'monitors'))
                if len(outputs) == 1 and outputs[0]['name'] == 'playback-test' and outputs[0]['width'] > 0:
                    break
                assert time.monotonic() < deadline
                time.sleep(.05)
            env['QT_QPA_PLATFORM'] = 'wayland'
            env.pop('QT_QUICK_BACKEND', None)
        bus = subprocess.Popen(['dbus-daemon', '--session', '--nofork', '--print-address'], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        children.append(bus)
        env['DBUS_SESSION_BUS_ADDRESS'] = bus.stdout.readline().strip()
        media = root / 'clip with spaces.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=320x180:r=15:d=8',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000:duration=8',
                        '-c:v', 'mpeg4', '-c:a', 'aac', '-threads', '1', str(media)], check=True)
        audio = root / 'sound.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(media), '-vn', str(audio)], check=True)
        corrupt = root / 'broken.mp4'; corrupt.write_text('not a video')
        ui = root / 'ui'; ui.mkdir()
        source = Path(__file__).resolve().parents[1] / 'ui'
        for name in ['MediaPreview.qml', 'SearchSlider.qml', 'SearchButton.qml', 'SearchText.qml', 'SearchTheme.qml', 'qmldir']:
            shutil.copy(source / name, ui / name)
        (ui / 'Commons').symlink_to('/usr/share/omarchy/shell/Commons', target_is_directory=True)
        (ui / 'Ui').symlink_to('/usr/share/omarchy/shell/Ui', target_is_directory=True)
        qml = QML.replace('VIDEO_URL', media.as_uri()).replace('AUDIO_URL', audio.as_uri()).replace('BROKEN_URL', corrupt.as_uri())
        (ui / 'shell.qml').write_text(qml)
        panel = spawn(['quickshell', '--path', str(ui), '--no-duplicate', '--no-color'], 'ui')
        deadline = time.monotonic() + 45
        captured = set()
        while panel.poll() is None:
            assert time.monotonic() < deadline, (root / 'ui.log').read_text()
            output = (root / 'ui.log').read_text()
            for phase in ['VIDEO', 'NARROW', 'ERROR']:
                if args.native and f'CAPTURE_{phase}' in output and phase not in captured:
                    subprocess.run(['grim', str(root / (phase.lower() + '.png'))], env=env, check=True, capture_output=True)
                    captured.add(phase)
            time.sleep(.05)
        output = (root / 'ui.log').read_text()
        assert 'PLAYBACK_PASS' in output and 'FAIL!' not in output and 'Error:' not in output, output
        print('PASS: manual playback, matched timestamp, pause/seek/mute/volume, stale requests, switch/hide, audio, narrow controls and corrupt media')
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=8)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
        for log in logs: log.close()


QML = r'''import QtQuick
import QtQuick.Layouts
import QtTest
import Quickshell
ShellRoot {
    id: harness
    QtObject {
        id: controller
        property var selectedAsset: ({assetId: "video", mediaKind: "video", startMs: 2000})
        property int previewGeneration: 0
        property bool ready: true
        property string previewUrl: ""
        property string secondaryText: "#bbbbbb"
        property var colors: ({background: "#141414", foreground: "#eeeeee", accent: "#a6c8e0", muted: "#aaaaaa", border: "#666666", error: "#ff8888"})
        property string url: "VIDEO_URL"
        property bool delay: false
        property var callback: null
        signal externalAppOpened()
        function request(command, done) { if (delay) callback = done; else done({url: url}); }
    }
    FloatingWindow {
        id: window
        visible: true
        color: "#141414"
        implicitWidth: 620
        implicitHeight: 440
        MediaPreview { id: preview; property int testWidth: 600; anchors.centerIn: parent; width: Math.min(testWidth, window.width - 40); height: Math.min(400, window.height - 40); app: controller; active: window.visible }
        TestCase {
            name: "Playback"
            when: window.visible
            function button(name) { const b = findChild(preview, name); verify(b !== null, name); return b; }
            function choose(kind, url, start) {
                controller.url = url;
                controller.selectedAsset = {assetId: kind + controller.previewGeneration, mediaKind: kind, startMs: start};
                ++controller.previewGeneration;
            }
            function test_playback() {
                wait(300);
                // Keep synthetic tones inaudible while exercising the actual audio decoder.
                mouseClick(button("playbackMute"));
                verify(!preview.loaded && !preview.playing);
                mouseClick(button("playbackToggle"));
                tryCompare(preview, "playing", true, 10000);
                verify(preview.position >= 1900 && preview.position < 4000, "start at matched segment");
                wait(500); console.log("CAPTURE_VIDEO"); wait(700);
                mouseClick(button("playbackToggle"));
                tryCompare(preview, "playing", false);
                const paused = preview.position; wait(200); compare(preview.position, paused);
                const seek = button("playbackSeek"); mouseClick(seek, seek.width * 0.7, seek.height / 2);
                verify(preview.position > 4500, "seek control");
                const volume = button("playbackVolume"); mouseClick(volume, volume.width * 0.3, volume.height / 2);
                verify(volume.value < 0.5, "volume control");
                choose("audio", "AUDIO_URL", 1000);
                verify(!preview.loaded && !preview.playing, "selection releases old file");
                mouseClick(button("playbackToggle")); tryCompare(preview, "playing", true, 10000);
                verify(preview.position >= 900);
                preview.testWidth = 300; wait(300);
                verify(button("playbackVolume").width > 20);
                console.log("CAPTURE_NARROW"); wait(700);
                preview.active = false; verify(!preview.playing && !preview.loaded);
                preview.active = true; verify(!preview.playing);
                controller.delay = true;
                mouseClick(button("playbackToggle")); verify(preview.loading);
                const late = controller.callback;
                choose("audio", "AUDIO_URL", 0); late({url: "VIDEO_URL"});
                verify(!preview.loaded && !preview.loading, "ignore stale source reply");
                controller.delay = false;
                choose("audio", "https://example.invalid/media.wav", 0);
                mouseClick(button("playbackToggle"));
                verify(preview.playbackError.length > 0 && !preview.loaded, "reject remote media URLs");
                choose("audio", "AUDIO_URL", 60000);
                mouseClick(button("playbackToggle"));
                tryVerify(() => preview.playbackError.includes("timestamp"), 10000);
                verify(!preview.playing, "invalid matched timestamp must not play from zero");
                choose("video", "BROKEN_URL", 0);
                mouseClick(button("playbackToggle"));
                tryVerify(() => preview.playbackError.length > 0, 10000);
                verify(!preview.playing && !preview.loading);
                console.log("CAPTURE_ERROR"); wait(700);
                choose("audio", "AUDIO_URL", 0);
                mouseClick(button("playbackToggle")); tryCompare(preview, "playing", true, 10000);
                controller.externalAppOpened(); verify(!preview.playing && !preview.loaded);
                console.log("PLAYBACK_PASS");
            }
            function cleanupTestCase() { Qt.quit(); }
        }
    }
}
'''

if __name__ == '__main__':
    main()
