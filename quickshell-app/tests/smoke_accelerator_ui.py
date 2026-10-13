#!/usr/bin/env python3
"""Check accelerator status semantics without models, GPU inference or live data."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    source = Path(__file__).resolve().parents[1] / 'ui'
    with tempfile.TemporaryDirectory(prefix='fractal-accelerator-') as directory:
        root = Path(directory)
        for name in ['AcceleratorSection.qml', 'SearchText.qml', 'SearchButton.qml', 'SearchTheme.qml', 'qmldir']:
            shutil.copy(source / name, root / name)
        (root / 'Commons').symlink_to('/usr/share/omarchy/shell/Commons', target_is_directory=True)
        (root / 'Ui').symlink_to('/usr/share/omarchy/shell/Ui', target_is_directory=True)
        (root / 'shell.qml').write_text('''import QtQuick
import Quickshell
ShellRoot {
    QtObject {
        id: diagnostic
        property bool checking: false
        property var report: null
        property string errorMessage: ""
        property string details: "Fixture requirements"
        property string summary: "Fixture summary"
        property int refreshCount: 0
        function refresh() { refreshCount++; }
    }
    QtObject {
        id: app
        property var runtimeDiagnostics: diagnostic
        property bool ready: true
        property bool busy: false
        property int statusRefreshCount: 0
        function refreshStatus() { statusRefreshCount++; }
        property string configuredProvider: "cpu"
        property var status: ({provider: "CPU", textModelLoaded: true})
        property var colors: ({foreground: "#eeeeee", background: "#111111", accent: "#88ffaa", error: "#ff8888"})
        property color secondaryText: "#bbbbbb"
        property color borderColor: "#444444"
    }
    FloatingWindow {
        visible: true
        implicitWidth: 540
        implicitHeight: 540
        AcceleratorSection { id: card; width: 520; app: app }
    }
    Timer {
        running: true
        interval: 100
        onTriggered: {
            function check(condition, message) { if (!condition) throw new Error(message); }
            try {
                check(diagnostic.refreshCount === 1, "Expected one automatic check");
                check(app.statusRefreshCount === 1, "Expected one visible runtime status refresh");
                check(!card.expanded, "Details should start collapsed");
                diagnostic.report = {checksPassed: false};
                check(card.cpuActive && !card.gpuAvailable && card.gpuState === "Unavailable", "Missing CUDA must leave actual CPU active");
                diagnostic.report = {checksPassed: true};
                check(card.gpuAvailable && card.cpuActive && !card.gpuActive && card.gpuState === "Available", "Capability must not imply active GPU");
                app.configuredProvider = "cuda";
                app.ready = false;
                check(!card.cpuActive && !card.gpuActive, "Disconnected CUDA must not claim CPU fallback");
                app.status = {provider: "CUDA", textModelLoaded: true};
                app.ready = true;
                check(card.gpuActive && !card.cpuActive && card.gpuState === "Active", "Loaded CUDA must show GPU active");
                diagnostic.errorMessage = "Fixture probe failed";
                check(!card.gpuAvailable && card.diagnosticText.includes("Fixture probe failed"), "Failed probe must remain visible and copyable");
                app.status = {provider: "CUDA", textModelLoaded: false};
                check(card.gpuState === "Check failed", "Diagnostic failure must not mean unavailable");
                diagnostic.errorMessage = "";
                diagnostic.report = null;
                diagnostic.checking = true;
                check(card.gpuState === "Checking…", "Pending check must be explicit");
                console.log("ACCELERATOR_PASS");
            } catch (error) { console.error("ACCELERATOR_FAIL " + error); }
            Qt.quit();
        }
    }
}''')
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software', QT_QUICK_CONTROLS_STYLE='Basic', QT_QPA_PLATFORMTHEME='generic')
        env.pop('WAYLAND_DISPLAY', None)
        result = subprocess.run(['quickshell', '--path', str(root), '--no-duplicate', '--no-color'], env=env, capture_output=True, text=True, timeout=15)
        output = result.stdout + result.stderr
        assert result.returncode == 0 and 'ACCELERATOR_PASS' in output and 'ACCELERATOR_FAIL' not in output, output
        print('PASS: CPU/missing CUDA, available versus active GPU, disconnected provider, diagnostic failures, collapsed details and automatic check.')


if __name__ == '__main__':
    main()
