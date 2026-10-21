#!/usr/bin/env python3
"""Open/close real viewers and resummon in a private nested Hyprland session.

Requires Hyprland, hyprctl, Quickshell, imv, grim, FFmpeg, pinned models and host
GPU/socket access. Only synthetic media and a temporary catalog are used; no
shared desktop input.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

from smoke_backend import png
from smoke_plugin import launch, stop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', type=Path)
    parser.add_argument('models', type=Path)
    parser.add_argument('--ui-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'ui',
                        help='UI source to validate with the matching backend')
    parser.add_argument('--chrome-only', action='store_true', help='Check panel geometry and actions without inference')
    parser.add_argument('--display-scale', type=float, choices=[1, 1.25], default=1)
    args = parser.parse_args()
    for command in ['Hyprland', 'hyprctl', 'quickshell', 'imv', 'grim', 'dbus-daemon', 'ffmpeg', 'ffprobe']:
        assert shutil.which(command), f'Missing test dependency: {command}'
    # Hyprland appends its build hash and timestamp to Unix socket paths.
    root = Path(tempfile.mkdtemp(prefix='fsr-', dir='/tmp'))
    print(f'Test artifacts: {root}', flush=True)
    runtime = root / 'run'; runtime.mkdir(mode=0o700)
    config = root / 'hyprland.lua'
    config.write_text('''hl.monitor({output="", mode="1280x900@60", position="auto", scale=1})
hl.config({input={follow_mouse=0},animations={enabled=false},misc={disable_hyprland_logo=true,disable_splash_rendering=true}})
'''.replace('scale=1}', 'scale=' + str(args.display_scale) + '}'))
    env = dict(os.environ, XDG_RUNTIME_DIR=str(runtime),
               QT_QPA_PLATFORM='wayland', QT_QUICK_CONTROLS_STYLE='Basic', QT_QPA_PLATFORMTHEME='gtk3',
               FRACTAL_SEARCH_UI_TEST='1', FRACTAL_RESUMMON_ROOT=str(root))
    for key in ['DISPLAY', 'WAYLAND_DISPLAY', 'HYPRLAND_INSTANCE_SIGNATURE', 'DBUS_SESSION_BUS_ADDRESS']:
        env.pop(key, None)
    # The nested compositor uses the host render device. Only its
    # startup inherits this socket; all test apps use its private socket below.
    env['WAYLAND_DISPLAY'] = str(Path(os.environ['XDG_RUNTIME_DIR']) / os.environ['WAYLAND_DISPLAY'])
    children = []
    logs = []
    viewer = None

    def spawn(command, name):
        log = (root / f'{name}.log').open('w'); logs.append(log)
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        children.append(process)
        return process

    def close_viewer():
        nonlocal viewer
        if viewer is not None:
            os.kill(viewer, signal.SIGTERM)
            viewer = None
            (root / 'viewer.pid').unlink()

    try:
        compositor = spawn(['Hyprland', '-c', str(config)], 'hyprland')
        deadline = time.monotonic() + 15
        while not list(runtime.glob('wayland-*.lock')):
            assert compositor.poll() is None, (root / 'hyprland.log').read_text()
            assert time.monotonic() < deadline, 'Nested compositor did not start'
            time.sleep(.05)
        sockets = list(runtime.glob('wayland-*.lock'))
        assert len(sockets) == 1
        env['WAYLAND_DISPLAY'] = sockets[0].name.removesuffix('.lock')
        signatures = list((runtime / 'hypr').iterdir()); assert len(signatures) == 1
        env['HYPRLAND_INSTANCE_SIGNATURE'] = signatures[0].name
        # The socket can appear before the nested output is configured. Starting
        # Qt then creates a placeholder screen and can strand the layer surface
        # at its provisional size, so wait for actual compositor output geometry.
        deadline = time.monotonic() + 15
        while True:
            assert compositor.poll() is None, (root / 'hyprland.log').read_text()
            probe = subprocess.run(['hyprctl', '-j', 'monitors'], env=env,
                                   capture_output=True, text=True, timeout=3)
            monitors = json.loads(probe.stdout) if probe.returncode == 0 else []
            if any(monitor.get('width', 0) > 0 and monitor.get('height', 0) > 0 for monitor in monitors):
                break
            assert time.monotonic() < deadline, 'Nested compositor has no configured output'
            time.sleep(.05)
        # A nested Wayland output follows its host window size. Move the private
        # test session to a fixed headless output before starting Qt, so desktop
        # tiling cannot crop captures or invalidate click coordinates mid-test.
        nested_outputs = [monitor['name'] for monitor in monitors]
        subprocess.run(['hyprctl', 'output', 'create', 'headless', 'fractal-test'],
                       env=env, check=True, capture_output=True, timeout=5)
        for output in nested_outputs:
            subprocess.run(['hyprctl', 'output', 'remove', output],
                           env=env, check=True, capture_output=True, timeout=5)
        deadline = time.monotonic() + 10
        while True:
            probe = subprocess.run(['hyprctl', '-j', 'monitors'], env=env,
                                   capture_output=True, text=True, check=True, timeout=3)
            monitors = json.loads(probe.stdout)
            if len(monitors) == 1 and monitors[0]['name'] == 'fractal-test' and monitors[0]['width'] == 1280 and monitors[0]['height'] == 900 and abs(monitors[0]['scale'] - args.display_scale) < .001:
                break
            assert time.monotonic() < deadline, monitors
            time.sleep(.05)
        (root / 'monitors.json').write_text(json.dumps(monitors))
        # Activated desktop services must inherit the private compositor too.
        bus = subprocess.Popen(['dbus-daemon', '--session', '--nofork', '--print-address'],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        children.append(bus)
        env['DBUS_SESSION_BUS_ADDRESS'] = bus.stdout.readline().strip()
        assert env['DBUS_SESSION_BUS_ADDRESS'].startswith('unix:')

        # A private FileManager1 command fixture records reveal requests.
        backend_commands = root / 'backend-commands'; backend_commands.mkdir()
        file_manager = backend_commands / 'gdbus'
        file_manager.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
assert 'org.freedesktop.FileManager1.ShowItems' in sys.argv
Path(os.environ['FRACTAL_RESUMMON_ROOT'], 'reveal.json').write_text(json.dumps(sys.argv[1:]))
print('()')
''')
        file_manager.chmod(0o755)
        # Service management stays inside this private session. The fixture
        # asks this test to restart its own backend and never calls systemd.
        systemctl = backend_commands / 'systemctl'
        systemctl.write_text("""#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
root = Path(os.environ['FRACTAL_RESUMMON_ROOT'])
assert sys.argv[1] == '--user' and sys.argv[3] == 'fractal-search.service', sys.argv
operation = sys.argv[2]
if operation == 'is-enabled':
    print('disabled'); sys.exit(1)
if operation == 'start':
    sys.exit(0)
assert operation == 'restart', 'Refusing non-test service operation'
if (root / 'restart.fail').exists():
    print('Private restart fixture failed', file=sys.stderr); sys.exit(7)
(root / 'restart.request').write_text('restart')
deadline = time.monotonic() + 20
while not (root / 'restart.complete').exists():
    assert time.monotonic() < deadline, 'Private restart was not completed'
    time.sleep(.02)
""")
        systemctl.chmod(0o755)
        env['PATH'] = str(backend_commands) + ':' + env['PATH']
        backend_root = root / 'backend'; backend_root.mkdir()
        backend, client, socket, log = launch(args.binary.resolve(), args.models.resolve(), backend_root, env)
        children.append(backend); logs.append(log)
        client.close()
        env['FRACTAL_SEARCH_SOCKET'] = str(socket)
        env['FRACTAL_SEARCH_BINARY'] = str(args.binary.resolve())
        ui = root / 'ui'
        shutil.copytree(args.ui_dir.resolve(), ui)
        (ui / 'Commons').symlink_to('/usr/share/omarchy/shell/Commons', target_is_directory=True)
        (ui / 'Ui').symlink_to('/usr/share/omarchy/shell/Ui', target_is_directory=True)
        # Test-only controls dispatch input exclusively to this private window.
        settings_source = ui / 'SettingsPage.qml'
        settings_source.write_text(settings_source.read_text().replace('id: page', 'id: page; property alias testScroll: settingsScroll').replace('text: modelData', 'objectName: modelData; text: modelData').replace('text: "Indexing issues"', 'objectName: "indexingIssuesButton"; text: "Indexing issues"').replace('text: "Keep folders up to date"', 'objectName: "watchChangesToggle"; text: "Keep folders up to date"'))
        panel_source = ui / 'SearchPanel.qml'
        panel_source.write_text(panel_source.read_text().replace('import QtQuick\n', 'import QtQuick\nimport QtTest\n').replace('id: panel', 'id: panel; TestCase { id: input; when: false }', 1).replace('ResultActions {', 'ResultActions { id: previewTestActions', 1).replace(
            'function refreshDiagnostics()',
            'function scrollSettings(amount: real): void { settingsPage.testScroll.contentItem.contentY = Math.min(amount, Math.max(0, settingsPage.testScroll.contentItem.contentHeight - settingsPage.testScroll.contentItem.height)); }\n        function clickTab(name: string): void { const button = input.findChild(settingsPage, name); if (!button) throw new Error("Missing tab " + name); input.mouseClick(button); }\n        function clickOpen(): void { const button = input.findChild(previewTestActions, "openImageButton"); if (!button) throw new Error("Missing open action"); input.mouseClick(button); }\n        function tabKey(key: int): void { input.keyClick(key); }\n        function clickReveal(): void { input.mouseClick(input.findChild(previewTestActions, "revealImageButton")); }\n        function toggleWatch(): void { input.mouseClick(input.findChild(settingsPage, "watchChangesToggle")); }\n        function chooseImageBatch(size: int): void { const choice = input.findChild(settingsPage, "imageBatchSizeChoice"); if (!choice) throw new Error("Missing image batch size"); input.mouseClick(choice); input.keyClick(Qt.Key_Home); for (let n = 1; n < size; ++n) input.keyClick(Qt.Key_Down); input.keyClick(Qt.Key_Return); }\n        function clickRestartNow(): void { input.mouseClick(restartDialog.confirmButton); }\n        function refreshDiagnostics()'))
        panel_source.write_text(panel_source.read_text().replace('query: app.query,', 'query: app.query, mediaKind: app.mediaKind, previewText: app.previewText(),').replace('function refreshDiagnostics()', '''function chooseFilter(name: string, index: int): void {
            const choice = input.findChild(canvas, name);
            if (!choice) throw new Error("Missing filter " + name);
            input.mouseClick(choice); input.keyClick(Qt.Key_Home);
            for (let n = 0; n < index; ++n) input.keyClick(Qt.Key_Down);
            input.keyClick(Qt.Key_Return);
        }
        function typeQuery(text: string): void {
            searchField.forceActiveFocus(); input.keyClick(Qt.Key_A, Qt.ControlModifier);
            if (!text.length) input.keyClick(Qt.Key_Backspace);
            for (let n = 0; n < text.length; ++n) input.keyClick(text.charAt(n));
            input.keyClick(Qt.Key_Return);
        }
        function queryAndLoading(text: string): bool {
            app.search(text);
            return app.loading;
        }
        function refreshDiagnostics()'''))
        panel_source.write_text(panel_source.read_text().replace('function refreshDiagnostics()', '''function partialVideoReport(): void {
            app.restoreReport({failed: 1, failures: [{path: "/fixture/movie.mp4", error: "Visuals and valid audio indexed; 1 audio segment(s) skipped. Rescan to retry missing audio. Audio segment 1 (30021–30037 ms): Audio decoder returned 0 samples"}], failuresOmitted: 0});
        }
        function clickIssueClose(): void { input.mouseClick(issueClose); }
        function clickIndexingIssues(): void {
            const button = input.findChild(settingsPage, "indexingIssuesButton");
            if (!button) throw new Error("Missing indexing issues button");
            input.mouseClick(button);
        }
        function uiScale(value: real): void { Style.spacingScale = value; }
        function footerAction(index: int): void { input.mouseClick(hotkeysFlow.children[index]); }
        function filterGeometry(): string {
            const kind = input.findChild(canvas, "mediaKindFilter");
            const folder = input.findChild(canvas, "sourceFilter");
            return JSON.stringify({
                typeLabel: kind.displayText, summary: resultsSummary.text,
                typeY: kind.mapToItem(canvas, 0, 0).y, folderY: folder.mapToItem(canvas, 0, 0).y,
                typeWidth: kind.width, folderWidth: folder.width,
                typeX: kind.mapToItem(canvas, 0, 0).x, folderX: folder.mapToItem(canvas, 0, 0).x,
                previewVisible: previewPane.visible, padding: Style.space(16),
                listX: resultsList.mapToItem(canvas, 0, 0).x, listWidth: resultsList.width,
                previewX: previewPane.mapToItem(canvas, 0, 0).x, previewWidth: previewPane.width,
                typeHeight: kind.height, folderHeight: folder.height,
                summaryY: resultsSummary.mapToItem(canvas, 0, 0).y
            });
        }
        function chromeGeometry(): string {
            return JSON.stringify({
                panelX: canvas.x, panelY: canvas.y, panelWidth: canvas.width,
                panelHeight: canvas.height, footerY: hotkeysFooter.y,
                footerHeight: hotkeysFooter.height, bottomInset: canvas.bottomInset,
                controls: Array.from(hotkeysFlow.children).filter(item => item.visible).map(item => {
                    const point = item.mapToItem(canvas, 0, 0);
                    return {x: point.x, y: point.y, width: item.width, height: item.height};
                })
            });
        }
        function refreshDiagnostics()'''))
        panel_source.write_text(panel_source.read_text().replace('previewLoading: app.previewLoading,',
            'previewLoading: app.previewLoading, mediaPlaying: mediaPreview.playing, mediaLoaded: mediaPreview.loaded, mediaError: mediaPreview.playbackError,').replace('function refreshDiagnostics()',
            'function clickPlayback(): void { input.mouseClick(input.findChild(mediaPreview, "playbackToggle")); }\n        function mutePlayback(): void { input.mouseClick(input.findChild(mediaPreview, "playbackMute")); }\n        function refreshDiagnostics()'))
        # An opaque desktop window exercises remapping above another app.
        shell = ui / 'shell.qml'
        shell.write_text(shell.read_text().replace('ShellRoot {', '''ShellRoot {
    FloatingWindow { visible: true; title: "Fractal test backdrop"; color: "#f4e9d8"; implicitWidth: 1280; implicitHeight: 900 }
'''))
        commands = root / 'commands'; commands.mkdir()
        opener = commands / 'gio'
        opener.write_text('''#!/usr/bin/env python3
import os, subprocess, sys
from pathlib import Path
from urllib.parse import unquote, urlsplit
root = Path(os.environ['FRACTAL_RESUMMON_ROOT'])
if (root / 'opener.fail').exists():
    print('Private default application launch failed', file=sys.stderr)
    sys.exit(17)
assert sys.argv[1] == 'open' and len(sys.argv) == 3
url = urlsplit(sys.argv[2]); assert url.scheme == 'file' and not url.netloc
path = Path(unquote(url.path)); assert path.is_relative_to(root / 'photos')
with (root / 'viewer.log').open('a') as log:
    viewer = subprocess.Popen(['imv', str(path)], stdout=log, stderr=log)
(root / 'viewer.pid').write_text(str(viewer.pid))
''')
        opener.chmod(0o755)
        env['PATH'] = str(commands) + ':' + env['PATH']
        panel = spawn(['quickshell', '--path', str(ui), '--no-duplicate', '--no-color'], 'ui')

        def call(target, method, *arguments):
            result = subprocess.run(['quickshell', 'ipc', '--path', str(ui), 'call', '--', target, method, *arguments],
                                    env=env, text=True, capture_output=True, timeout=3)
            assert result.returncode == 0 and not result.stdout.startswith(('Too many arguments', 'Not enough arguments', 'No such')), result.stdout + result.stderr + (root / 'ui.log').read_text()
            return result.stdout.strip()

        def wait(predicate, timeout=10):
            deadline = time.monotonic() + timeout
            state = None
            while time.monotonic() < deadline:
                assert panel.poll() is None, (root / 'ui.log').read_text()
                state = json.loads(call('test', 'state'))
                if predicate(state):
                    return state
                time.sleep(.02)
            raise AssertionError(state)

        deadline = time.monotonic() + 15
        while not list((runtime / 'quickshell/by-id').glob('*/ipc.sock')):
            assert panel.poll() is None, (root / 'ui.log').read_text()
            assert time.monotonic() < deadline, (root / 'ui.log').read_text()
            time.sleep(.05)
        wait(lambda s: s['ready'] and not s['loading'] and s['panelVisible'])
        for scale in [1, 1.5]:
            call('test', 'uiScale', str(scale))
            time.sleep(.4)
            state = wait(lambda s: not s['animating'] and s['panelHeight'] == s['compactHeight'])
            chrome = json.loads(call('test', 'chromeGeometry'))
            for control in chrome['controls']:
                assert control['y'] >= chrome['footerY'], chrome
                assert control['y'] + control['height'] < chrome['panelHeight'] - chrome['bottomInset'], chrome
                assert control['x'] >= 0 and control['x'] + control['width'] < state['panelWidth'], chrome
            (root / f'compact-{scale}.json').write_text(json.dumps(chrome))
            subprocess.run(['grim', str(root / f'compact-{scale}.png')], env=env, check=True, timeout=5)
        call('test', 'uiScale', '1')
        call('test', 'footerAction', '2')  # Click Settings hint.
        wait(lambda s: s['page'] == 'settings')
        call('test', 'footerAction', '3')  # Click Back hint.
        wait(lambda s: s['page'] == 'summon')
        call('test', 'footerAction', '1')  # Click Search hint on the empty catalog.
        wait(lambda s: s['page'] == 'results' and not s['loading'] and s['count'] == 0)
        time.sleep(.3)
        subprocess.run(['grim', str(root / 'empty.png')], env=env, check=True, timeout=5)

        def check_filters(summary, label):
            geometry = json.loads(call('test', 'filterGeometry'))
            assert geometry['typeLabel'] == label and geometry['summary'] == summary, geometry
            assert geometry['typeY'] == geometry['folderY'], geometry
            assert geometry['typeHeight'] == geometry['folderHeight'], geometry
            if geometry['previewVisible']:
                assert abs(geometry['typeX'] - geometry['listX']) <= 1, geometry
                assert abs(geometry['typeWidth'] - geometry['listWidth']) <= 1, geometry
                assert abs(geometry['folderX'] - geometry['previewX'] + geometry['padding']) <= 1, geometry
                assert abs(geometry['folderWidth'] - geometry['previewWidth'] - 2 * geometry['padding']) <= 1, geometry
            else:
                assert abs(geometry['typeWidth'] - geometry['folderWidth']) <= 1, geometry
            assert geometry['summaryY'] > geometry['typeY'] + geometry['typeHeight'], geometry

        check_filters('No results', 'All file types')
        if args.chrome_only:
            call('test', 'settings')
            wait(lambda s: s['page'] == 'settings' and not s['animating'])
            call('test', 'partialVideoReport')
            state = wait(lambda s: s['failed'] == 1)
            assert 'file(s) with indexing issues' in state['error'], state
            call('test', 'scrollSettings', '10000')
            time.sleep(.3)
            call('test', 'clickIndexingIssues')
            wait(lambda s: s['skippedFilesVisible'])
            time.sleep(.3)
            subprocess.run(['grim', str(root / 'partial-video-report.png')], env=env, check=True, timeout=5)
            call('test', 'clickIssueClose')
            wait(lambda s: not s['skippedFilesVisible'])
            call('test', 'resize', '480', '440')
            time.sleep(.4)
            subprocess.run(['grim', str(root / 'chrome-narrow.png')], env=env, check=True, timeout=5)
            call('test', 'scrollSettings', '10000')
            time.sleep(.3)
            call('test', 'clickIndexingIssues')
            wait(lambda s: s['skippedFilesVisible'])
            time.sleep(.3)
            subprocess.run(['grim', str(root / 'partial-video-report-narrow.png')], env=env, check=True, timeout=5)
            call('test', 'clickIssueClose')
            wait(lambda s: not s['skippedFilesVisible'])
            call('test', 'back')
            call('search', 'toggle')
            call('search', 'summon')
            wait(lambda s: s['visible'] and s['panelVisible'] and s['active'])
            call('search', 'close')
            assert panel.wait(timeout=20) == 0
            assert backend.wait(timeout=20) == 0
            print(f'Panel chrome passed on a light backdrop at display scale {args.display_scale}: compact/scaled spacing, footer clicks, empty results, normal/narrow partial-video reports, report button clicks, narrow settings and resummon.', flush=True)
            return
        photos = root / 'photos'; photos.mkdir()
        png(photos / 'red.png', color=(200, 40, 20))
        png(photos / 'blue.png', color=(20, 40, 200))
        call('test', 'index', str(photos))
        wait(lambda s: s['indexing'], timeout=10)
        wait(lambda s: not s['indexing'] and s['assetCount'] == 2, timeout=60)
        # Observe admission synchronously: a warmed GPU query can complete
        # before the next separate IPC snapshot reaches the main QML thread.
        assert call('test', 'queryAndLoading', 'red image') == 'true'
        wait(lambda s: not s['loading'] and s['count'] == 2 and bool(s['previewUrl']), timeout=60)

        call('test', 'acknowledge')
        time.sleep(.3)
        subprocess.run(['grim', str(root / 'results.png')], env=env, check=True, timeout=5)
        check_filters('2 results', 'All file types')
        selected = json.loads(call('test', 'state'))['selectedAsset']
        call('test', 'clickReveal')
        deadline = time.monotonic() + 5
        while not (root / 'reveal.json').exists():
            assert time.monotonic() < deadline, 'Show in folder did not dispatch'
            time.sleep(.02)
        arguments = json.loads((root / 'reveal.json').read_text())
        assert json.loads(arguments[-2]) == [Path(selected['path']).as_uri()], arguments
        call('search', 'summon')
        wait(lambda s: s['active'])
        call('test', 'settings')
        wait(lambda s: s['page'] == 'settings')
        time.sleep(.3)
        subprocess.run(['grim', str(root / 'folders.png')], env=env, check=True, timeout=5)
        before = json.loads(call('test', 'state'))['settings']['watchChanges']
        call('test', 'toggleWatch')
        wait(lambda s: not s['loading'] and s['settings']['watchChanges'] != before)
        call('test', 'toggleWatch')
        wait(lambda s: not s['loading'] and s['settings']['watchChanges'] == before)
        call('test', 'clickTab', 'Preferences')
        wait(lambda s: s['settingsTab'] == 1)
        call('test', 'tabKey', str(0x01000014))  # Right
        wait(lambda s: s['settingsTab'] == 2)
        call('test', 'tabKey', str(0x01000012))  # Left
        wait(lambda s: s['settingsTab'] == 1)
        time.sleep(.2)
        subprocess.run(['grim', str(root / 'preferences.png')], env=env, check=True, timeout=5)
        call('test', 'resize', '480', '440')
        time.sleep(.3)
        subprocess.run(['grim', str(root / 'narrow-preferences.png')], env=env, check=True, timeout=5)
        call('test', 'resize', '0', '0')
        call('test', 'clickTab', 'Performance')
        wait(lambda s: s['settingsTab'] == 2)
        call('test', 'refreshDiagnostics')
        diagnostic = wait(lambda s: not s['runtimeDiagnosticChecking'] and bool(s['runtimeDiagnosticDetails']), timeout=25)
        assert not diagnostic['runtimeDiagnosticError'], diagnostic
        assert 'Required libraries:' in diagnostic['runtimeDiagnosticDetails']
        assert 'cuDNN 9' in diagnostic['runtimeDiagnosticDetails']
        call('test', 'scrollSettings', '0')
        time.sleep(.2)
        subprocess.run(['grim', str(root / 'diagnostics-settings.png')], env=env, check=True, timeout=5)
        call('test', 'scrollSettings', '150')
        call('test', 'chooseImageBatch', '4')
        pending_restart = wait(lambda s: s['restartConfirmationVisible'] and s['restartLaterFocused'])
        assert pending_restart['settings']['imageBatchSize'] == 4 and pending_restart['activeImageBatchSize'] == 1, pending_restart
        time.sleep(.2)
        subprocess.run(['grim', str(root / 'batch-restart-prompt.png')], env=env, check=True, timeout=5)
        call('test', 'resize', '480', '440')
        time.sleep(.3)
        subprocess.run(['grim', str(root / 'batch-restart-narrow.png')], env=env, check=True, timeout=5)
        call('test', 'tabKey', str(0x01000004))  # Return chooses the focused Later action.
        deferred = wait(lambda s: not s['restartConfirmationVisible'] and s['batchRestartRequired'])
        assert deferred['activeImageBatchSize'] == 1 and backend.poll() is None
        call('test', 'manageTestService', 'true')
        (root / 'restart.fail').touch()
        call('test', 'requestRestart')
        wait(lambda s: s['restartConfirmationVisible'])
        call('test', 'clickRestartNow')
        failed = wait(lambda s: not s['restarting'] and 'Private restart fixture failed' in s['error'])
        assert failed['activeImageBatchSize'] == 1 and failed['batchRestartRequired']
        (root / 'restart.fail').unlink()
        call('test', 'acknowledge')
        call('test', 'requestRestart')
        wait(lambda s: s['restartConfirmationVisible'])
        call('test', 'clickRestartNow')
        deadline = time.monotonic() + 5
        while not (root / 'restart.request').exists():
            assert time.monotonic() < deadline, (root / 'ui.log').read_text()
            time.sleep(.02)
        stop(backend)
        assert backend.returncode == 0
        backend, client, socket, log = launch(args.binary.resolve(), args.models.resolve(), backend_root, env)
        children.append(backend); logs.append(log); client.close()
        (root / 'restart.complete').touch()
        # Reconnection retains the query, which can load the new model before
        # results return. Use the same bounded cold-search deadline as above.
        restarted = wait(lambda s: s['ready'] and not s['loading'] and s['activeImageBatchSize'] == 4 and not s['batchRestartRequired'], timeout=60)
        assert restarted['assetCount'] == 2 and restarted['visible']
        call('test', 'manageTestService', 'false')
        call('test', 'resize', '0', '0')
        print('Batch size choice, Later focus, narrow prompt, restart failure and safe restart passed', flush=True)
        call('test', 'scrollSettings', '0')
        call('test', 'back')

        latencies = []
        for hide in [True, False]:
            call('test', 'updateSettings', json.dumps({'closeAfterOpening': hide}))
            wait(lambda s: not s['loading'] and s['settings']['closeAfterOpening'] == hide)
            for _ in range(8):
                call('test', 'browse')
                wait(lambda s: not s['loading'] and s['selectedAsset'] and s['previewUrl'] and s['active'])
                call('test', 'clickOpen')
                deadline = time.monotonic() + 3
                while not (root / 'viewer.pid').exists():
                    assert time.monotonic() < deadline, (root / 'ui.log').read_text()
                    time.sleep(.02)
                viewer = int((root / 'viewer.pid').read_text())
                wait(lambda s: s['visible'] == (not hide) and not s['active'], timeout=2)
                time.sleep(.15)
                close_viewer()
                started = time.monotonic()
                call('search', 'summon')
                wait(lambda s: s['visible'] and s['panelVisible'] and s['active'], timeout=2)
                latencies.append(time.monotonic() - started)
            print(f'8 viewer cycles passed with hide={hide}; maximum summon time {max(latencies):.3f}s', flush=True)

        media = root / 'media'; media.mkdir()
        (media / 'notes.md').write_text('A red bicycle beside the cafe. Exact identifier QX42.', encoding='utf-8')
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                        'sine=frequency=440:sample_rate=16000:duration=3', str(media / 'tone.wav')],
                       check=True, capture_output=True, timeout=30)
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                        'color=c=blue:s=96x64:r=2:d=3', '-an', '-c:v', 'mpeg4', str(media / 'clip.mp4')],
                       check=True, capture_output=True, timeout=30)
        call('test', 'index', str(media))
        wait(lambda s: not s['indexing'] and s['assetCount'] == 5 and s['fileCount'] == 5, timeout=180)
        call('test', 'browse')
        all_media = wait(lambda s: not s['loading'] and s['count'] == 5)
        for index, kind, count in [(1, 'image', 2), (2, 'text', 1), (3, 'audio', 1), (4, 'video', 1), (0, None, 5)]:
            call('test', 'chooseFilter', 'mediaKindFilter', str(index))
            state = wait(lambda s: not s['loading'] and s['mediaKind'] == kind and s['count'] == count)
            if kind in ['text', 'audio']:
                assert not state['previewUrl'] and state['previewText'], state
            elif kind == 'video':
                assert state['previewUrl'], state
            assert not state['mediaPlaying'] and not state['mediaLoaded'], state
            if kind in ['audio', 'video']:
                if kind == 'audio':
                    call('test', 'mutePlayback')
                call('test', 'clickPlayback')
                wait(lambda s: s['mediaPlaying'] and not s['mediaError'], timeout=15)
                call('test', 'clickPlayback')
                wait(lambda s: not s['mediaPlaying'] and s['mediaLoaded'])
        media_source_index = 1 + next(index for index, source in enumerate(all_media['sources']) if source['directory'] == str(media))
        call('test', 'chooseFilter', 'sourceFilter', str(media_source_index))
        filtered = wait(lambda s: not s['loading'] and s['count'] == 3)
        assert next(source for source in filtered['sources'] if source['id'] == filtered['sourceId'])['directory'] == str(media)
        call('test', 'chooseFilter', 'mediaKindFilter', '2')
        call('test', 'typeQuery', 'red bicycle')
        call('test', 'typeQuery', 'QX42')
        hybrid = wait(lambda s: not s['loading'] and s['activeQuery'] == 'QX42' and s['count'] == 1)
        assert hybrid['selectedAsset']['hasHybridScore'], hybrid
        call('test', 'resize', '480', '440')
        time.sleep(.3)
        check_filters('1 result', 'Text')
        subprocess.run(['grim', str(root / 'narrow-hybrid-text.png')], env=env, check=True, timeout=5)
        call('test', 'chooseFilter', 'mediaKindFilter', '0')
        call('test', 'chooseFilter', 'sourceFilter', '0')
        call('test', 'typeQuery', '')
        wait(lambda s: not s['loading'] and s['count'] == 5)
        call('test', 'chooseFilter', 'mediaKindFilter', '1')
        wait(lambda s: not s['loading'] and s['count'] == 2)
        call('test', 'resize', '0', '0')
        print('Real mixed-media indexing, all media/folder filters, repeated native typing, hybrid labels and narrow results passed', flush=True)
        call('test', 'browse')
        selected = wait(lambda s: not s['loading'] and not s['previewLoading'] and s['previewUrl'] and s['selectedAsset'])['selectedAsset']
        (root / 'opener.fail').touch()
        call('test', 'clickOpen')
        wait(lambda s: 'Private default application launch failed' in s['error'] and s['visible'])
        assert not (root / 'viewer.pid').exists()
        subprocess.run(['grim', str(root / 'open-failure.png')], env=env, check=True, timeout=5)
        (root / 'opener.fail').unlink()
        call('test', 'acknowledge')
        Path(selected['path']).unlink()
        call('test', 'clickOpen')
        wait(lambda s: 'Original media file is missing' in s['error'] and s['visible'])
        call('test', 'acknowledge')
        call('search', 'toggle')
        call('search', 'summon')
        wait(lambda s: s['visible'] and s['panelVisible'] and s['active'])
        subprocess.run(['grim', str(root / 'resummoned.png')], env=env, check=True, timeout=5)
        call('search', 'close')
        assert panel.wait(timeout=20) == 0
        assert backend.wait(timeout=20) == 0
        print(f'16 viewer cycles passed with both checkbox states; maximum summon time {max(latencies):.3f}s. '
              'Empty/loading/search/preview, opener/missing-original errors and clean stop passed.', flush=True)
    finally:
        close_viewer()
        for process in reversed(children):
            stop(process)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
