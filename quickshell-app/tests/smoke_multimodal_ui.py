#!/usr/bin/env python3
"""Offscreen mixed-media Controller checks; no backend, models, originals or desktop input."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    source = Path(__file__).resolve().parents[1] / 'ui'
    with tempfile.TemporaryDirectory(prefix='fractal-media-ui-') as directory:
        root = Path(directory)
        for name in ['Controller.qml', 'RuntimeDiagnostics.qml', 'ServiceAutostart.qml', 'ResultActions.qml', 'SearchButton.qml', 'SearchText.qml', 'SearchTheme.qml', 'qmldir']:
            shutil.copy(source / name, root / name)
        (root / 'Commons').symlink_to('/usr/share/omarchy/shell/Commons', target_is_directory=True)
        (root / 'shell.qml').write_text('''import QtQuick
import Quickshell
ShellRoot {
    id: test
    property var previewCallback: null
    property int previewCalls: 0
    property var fixtures: [
        {assetId: "asset_text", filename: "notes.md", path: "/fixture/notes.md", mediaKind: "text", segmentIndex: 1, startMs: null, endMs: null, textExcerpt: "A café and a red bicycle.", thumbnailUrl: "", width: 0, height: 0, similarity: null},
        {assetId: "asset_audio", filename: "speech.wav", path: "/fixture/speech.wav", mediaKind: "audio", segmentIndex: 1, startMs: 30000, endMs: 45000, textExcerpt: null, thumbnailUrl: "", width: 0, height: 0, similarity: null},
        {assetId: "asset_video", filename: "clip.mp4", path: "/fixture/clip.mp4", mediaKind: "video", segmentIndex: 0, startMs: 0, endMs: 30000, textExcerpt: null, thumbnailUrl: "file:///fixture/cached-frame.webp", width: 96, height: 64, similarity: null},
        {assetId: "asset_image", filename: "image.png", path: "/fixture/image.png", mediaKind: "image", segmentIndex: 0, startMs: null, endMs: null, textExcerpt: null, thumbnailUrl: "file:///fixture/cached-image.webp", width: 96, height: 64, similarity: null}
    ]
    function check(condition, message) { if (!condition) throw new Error(message); }
    Controller {
        id: controller
        socketPath: "/fixture/no-socket"
        manageService: false
        ready: true
        function request(command, callback) {
            if (command.type === "browse") callback({total: 4, items: test.fixtures});
            else if (command.type === "status") callback(status);
            else if (command.type === "preview") {
                test.previewCalls++;
                test.previewCallback = callback;
            } else throw new Error("Unexpected backend request: " + command.type);
        }
    }
    FloatingWindow {
        visible: true
        implicitWidth: 340
        implicitHeight: 100
        ResultActions { id: actions; width: 320; app: controller }
    }
    Timer {
        running: true
        interval: 100
        onTriggered: {
            try {
                controller.browse(false);
                check(controller.results.count === 4 && !controller.errorMessage, "Mixed-media browse must accept empty nonvisual thumbnails");
                check(controller.mediaLabel(controller.selectedAsset) === "Text · passage 2", "Text passage label");
                check(controller.previewText() === "A café and a red bicycle." && !controller.previewUrl, "Text uses cached excerpt");
                controller.loadPreview();
                check(test.previewCalls === 0, "Text must not request image preview");
                controller.select(1);
                controller.loadPreview();
                check(controller.mediaLabel(controller.selectedAsset) === "Audio · 0:30–0:45", "Audio time range");
                check(controller.previewText().includes("default app") && !controller.previewUrl && test.previewCalls === 0, "Audio must not request image preview");
                controller.select(2);
                controller.loadPreview();
                check(controller.previewUrl === "file:///fixture/cached-frame.webp" && test.previewCalls === 0, "Video must use cached thumbnail only");
                controller.select(3);
                controller.loadPreview();
                check(test.previewCalls === 1 && controller.previewLoading, "Image preview request");
                controller.select(0);
                test.previewCallback({url: "file:///fixture/stale-preview.png"});
                check(!controller.previewUrl && !controller.previewLoading, "Stale image response must not replace text preview");
                let open = null;
                for (const child of actions.children) {
                    if (child.objectName === "openImageButton") open = child;
                }
                check(open && open.text === "Open file", "Generic whole-file action");
                finish.start();
            } catch (error) { console.error("MEDIA_UI_FAIL " + error); Qt.quit(); }
        }
    }
    Timer {
        id: finish
        interval: 200
        onTriggered: {
            try {
                check(test.previewCalls === 1 && !controller.errorMessage && !controller.previewUrl, "Stale preview timer must leave nonvisual media alone");
                console.log("MEDIA_UI_PASS");
            } catch (error) { console.error("MEDIA_UI_FAIL " + error); }
            Qt.quit();
        }
    }
}''')
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software', QT_QUICK_CONTROLS_STYLE='Basic', QT_QPA_PLATFORMTHEME='generic')
        env.pop('WAYLAND_DISPLAY', None)
        result = subprocess.run(['quickshell', '--path', str(root), '--no-duplicate', '--no-color'], env=env, capture_output=True, text=True, timeout=15)
        output = result.stdout + result.stderr
        assert result.returncode == 0 and 'MEDIA_UI_PASS' in output and 'MEDIA_UI_FAIL' not in output, output
        print('PASS: mixed-media result parsing, kind/time/passage labels, excerpts, cached video thumbnail, nonvisual preview suppression, stale image response and Open file label')


if __name__ == '__main__':
    main()
