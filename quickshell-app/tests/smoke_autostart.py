#!/usr/bin/env python3
"""Exercise the real QML login setting offscreen with isolated systemctl state."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main():
    with tempfile.TemporaryDirectory(prefix='fractal-search-autostart-', dir='/tmp') as temp:
        root = Path(temp)
        ui = root / 'ui'; ui.mkdir()
        source = Path(__file__).resolve().parents[1] / 'ui/ServiceAutostart.qml'
        shutil.copy(source, ui)
        (ui / 'shell.qml').write_text('''import QtQuick
import Quickshell
import Quickshell.Io
ShellRoot {
    id: root
    property string error: ""
    ServiceAutostart {
        id: setting
        serviceName: "fractal-search.service"
        onFailed: message => root.error = message
    }
    IpcHandler {
        target: "test"
        function refresh(): void { root.error = ""; setting.refresh(); }
        function enable(): void { root.error = ""; setting.setStartAtLogin(true); }
        function disable(): void { root.error = ""; setting.setStartAtLogin(false); }
        function state(): string { return JSON.stringify({known: setting.known, enabled: setting.startAtLogin, busy: setting.busy, error: root.error}); }
        function quit(): void { Qt.quit(); }
    }
}''')
        commands = root / 'commands'; commands.mkdir()
        state = root / 'enabled'; state.write_text('disabled')
        fail = root / 'fail'
        trace = root / 'trace'
        systemctl = commands / 'systemctl'
        systemctl.write_text('''#!/usr/bin/env python3
import os,sys,time
from pathlib import Path
r=Path(os.environ['AUTOSTART_TEST_ROOT'])
args=sys.argv[1:]
assert args[0]=='--user' and args[-1]=='fractal-search.service',args
with (r/'trace').open('a') as out: out.write(' '.join(args)+'\\n')
if args[1]=='is-enabled':
    state=(r/'enabled').read_text(); print(state); sys.exit(0 if state=='enabled' else 1)
time.sleep(.15)
if (r/'fail').exists(): print('injected permission failure',file=sys.stderr); sys.exit(9)
(r/'enabled').write_text('enabled' if args[1]=='enable' else 'disabled')
''')
        systemctl.chmod(0o755)
        env = dict(os.environ, PATH=str(commands)+':'+os.environ['PATH'], AUTOSTART_TEST_ROOT=str(root), QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software', QT_QUICK_CONTROLS_STYLE='Basic', QT_QPA_PLATFORMTHEME='')
        log = (root / 'ui.log').open('w')
        process = subprocess.Popen(['quickshell', '--path', str(ui), '--no-duplicate', '--no-color'], env=env, stdout=log, stderr=log)
        def call(method):
            result = subprocess.run(['quickshell', 'ipc', '--path', str(ui), 'call', 'test', method], env=env, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr + (root / 'ui.log').read_text()
            return result.stdout.strip()
        def wait(predicate):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                assert process.poll() is None, (root / 'ui.log').read_text()
                result = subprocess.run(['quickshell', 'ipc', '--path', str(ui), 'call', 'test', 'state'], env=env, capture_output=True, text=True)
                if result.returncode == 0:
                    value = json.loads(result.stdout)
                    if predicate(value): return value
                time.sleep(.05)
            raise AssertionError((root / 'ui.log').read_text())
        try:
            wait(lambda value: not value['busy'])
            call('refresh'); wait(lambda value: value['known'] and not value['enabled'] and not value['busy'])
            call('enable'); wait(lambda value: value['known'] and value['enabled'] and not value['busy'])
            # Re-reading the persistent unit state must survive UI reconnection.
            call('refresh'); wait(lambda value: value['enabled'] and not value['busy'])
            fail.touch(); call('disable')
            value = wait(lambda value: not value['busy'] and bool(value['error']))
            assert value['enabled'] and 'injected permission failure' in value['error'], value
            fail.unlink(); call('disable'); wait(lambda value: value['known'] and not value['enabled'] and not value['busy'])
            state.write_text('masked'); call('refresh')
            value = wait(lambda value: not value['busy'] and bool(value['error']))
            assert not value['known'] and 'masked' in value['error'], value
            assert '--now' not in trace.read_text()
            call('quit'); assert process.wait(timeout=10) == 0
        finally:
            if process.poll() is None:
                process.terminate(); process.wait(timeout=10)
            log.close()
    print('Offscreen login setting: enabled/disabled persistence, failed changes, invalid state and no service stop passed')


if __name__ == '__main__':
    main()
