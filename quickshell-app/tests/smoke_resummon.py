#!/usr/bin/env python3
"""Open/close real viewers and resummon in a private nested Hyprland session.

Requires Hyprland, Quickshell, imv, grim, pinned models and host GPU/socket access.
Only synthetic photos and a temporary catalog are used; no shared desktop input.
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
    args = parser.parse_args()
    for command in ['Hyprland', 'quickshell', 'imv', 'grim', 'dbus-daemon']:
        assert shutil.which(command), f'Missing test dependency: {command}'
    # Hyprland appends its build hash and timestamp to Unix socket paths.
    root = Path(tempfile.mkdtemp(prefix='fsr-', dir='/tmp'))
    print(f'Test artifacts: {root}', flush=True)
    runtime = root / 'run'; runtime.mkdir(mode=0o700)
    config = root / 'hyprland.lua'
    config.write_text('''hl.monitor({output="", mode="1280x900@60", position="auto", scale=1})
hl.config({input={follow_mouse=0},animations={enabled=false},misc={disable_hyprland_logo=true,disable_splash_rendering=true}})
''')
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
        env['PATH'] = str(backend_commands) + ':' + env['PATH']
        backend_root = root / 'backend'; backend_root.mkdir()
        backend, client, socket, log = launch(args.binary.resolve(), args.models.resolve(), backend_root, env)
        children.append(backend); logs.append(log)
        client.close()
        env['FRACTAL_SEARCH_SOCKET'] = str(socket)
        env['FRACTAL_SEARCH_BINARY'] = str(args.binary.resolve())
        ui = root / 'ui'
        shutil.copytree(Path(__file__).resolve().parents[1] / 'ui', ui)
        (ui / 'Commons').symlink_to('/usr/share/omarchy/shell/Commons', target_is_directory=True)
        # Test-only controls dispatch input exclusively to this private window.
        settings_source = ui / 'SettingsPage.qml'
        settings_source.write_text(settings_source.read_text().replace('id: page', 'id: page; property alias testScroll: settingsScroll').replace('text: modelData', 'objectName: modelData; text: modelData').replace('text: "Keep folders up to date"', 'objectName: "watchChangesToggle"; text: "Keep folders up to date"'))
        panel_source = ui / 'SearchPanel.qml'
        panel_source.write_text(panel_source.read_text().replace('import QtQuick\n', 'import QtQuick\nimport QtTest\n').replace('id: panel', 'id: panel; TestCase { id: input; when: false }', 1).replace('ResultActions {', 'ResultActions { id: previewTestActions', 1).replace(
            'function refreshDiagnostics()',
            'function scrollSettings(amount: real): void { settingsPage.testScroll.contentItem.contentY = amount; }\n        function clickTab(name: string): void { const button = input.findChild(settingsPage, name); if (!button) throw new Error("Missing tab " + name); input.mouseClick(button); }\n        function clickOpen(): void { const button = input.findChild(previewTestActions, "openImageButton"); if (!button) throw new Error("Missing open action"); input.mouseClick(button); }\n        function tabKey(key: int): void { input.keyClick(key); }\n        function clickReveal(): void { input.mouseClick(input.findChild(previewTestActions, "revealImageButton")); }\n        function toggleWatch(): void { input.mouseClick(input.findChild(settingsPage, "watchChangesToggle")); }\n        function refreshDiagnostics()'))
        # An opaque desktop window exercises remapping above another app.
        shell = ui / 'shell.qml'
        shell.write_text(shell.read_text().replace('ShellRoot {', '''ShellRoot {
    FloatingWindow { visible: true; title: "Fractal test backdrop"; color: "#172030"; implicitWidth: 1280; implicitHeight: 900 }
'''))
        commands = root / 'commands'; commands.mkdir()
        opener = commands / 'xdg-open'
        opener.write_text('''#!/usr/bin/env python3
import os, subprocess, sys
from pathlib import Path
from urllib.parse import unquote, urlsplit
root = Path(os.environ['FRACTAL_RESUMMON_ROOT'])
url = urlsplit(sys.argv[1]); assert url.scheme == 'file' and not url.netloc
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
            assert result.returncode == 0, result.stderr + (root / 'ui.log').read_text()
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
        call('test', 'browse')
        wait(lambda s: s['page'] == 'results' and not s['loading'] and s['count'] == 0)
        photos = root / 'photos'; photos.mkdir()
        png(photos / 'red.png', color=(200, 40, 20))
        png(photos / 'blue.png', color=(20, 40, 200))
        call('test', 'index', str(photos))
        wait(lambda s: s['indexing'], timeout=10)
        wait(lambda s: not s['indexing'] and s['imageCount'] == 2, timeout=60)
        call('search', 'query', 'red image')
        wait(lambda s: s['loading'])
        wait(lambda s: not s['loading'] and s['count'] == 2 and bool(s['previewUrl']), timeout=60)

        call('test', 'acknowledge')
        time.sleep(.3)
        subprocess.run(['grim', str(root / 'results.png')], env=env, check=True, timeout=5)
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
        call('test', 'browse')
        selected = wait(lambda s: not s['loading'] and not s['previewLoading'] and s['previewUrl'] and s['selectedAsset'])['selectedAsset']
        Path(selected['path']).unlink()
        call('test', 'clickOpen')
        wait(lambda s: 'Original image is missing' in s['error'] and s['visible'])
        call('test', 'acknowledge')
        call('search', 'toggle')
        call('search', 'summon')
        wait(lambda s: s['visible'] and s['panelVisible'] and s['active'])
        subprocess.run(['grim', str(root / 'resummoned.png')], env=env, check=True, timeout=5)
        call('search', 'close')
        assert panel.wait(timeout=20) == 0
        assert backend.wait(timeout=20) == 0
        print(f'16 viewer cycles passed with both checkbox states; maximum summon time {max(latencies):.3f}s. '
              'Empty/loading/search/preview, missing-original error and clean stop passed.', flush=True)
    finally:
        close_viewer()
        for process in reversed(children):
            stop(process)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
