#!/usr/bin/env python3
"""GIO MIME/terminal dispatch and QML failures; private handlers and temporary files only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main():
    source = Path(__file__).resolve().parents[1] / 'ui' / 'DesktopFileOpener.qml'
    with tempfile.TemporaryDirectory(prefix='fractal-file-opener-', dir='/tmp') as temporary:
        root = Path(temporary)
        config, data, commands, ui = [root / name for name in ['config', 'data', 'commands', 'ui']]
        for directory in [config, data / 'applications', commands, ui]:
            directory.mkdir(parents=True)
        record = root / 'launch.json'
        handler = commands / 'handler'
        handler.write_text('''#!/usr/bin/python3
import json, os, sys
from pathlib import Path
Path(os.environ['OPEN_RECORD']).write_text(json.dumps({'args': sys.argv[1:], 'terminal': os.environ.get('OPEN_TERMINAL') == '1'}))
''')
        terminal = commands / 'xdg-terminal-exec'
        terminal.write_text('''#!/usr/bin/python3
import os, sys
args = sys.argv[1:]
if args and args[0] == '--':
    args = args[1:]
os.environ['OPEN_TERMINAL'] = '1'
os.execvp(args[0], args)
''')
        for executable in [handler, terminal]:
            executable.chmod(0o755)
        for name, mime, use_terminal in [('text', 'text/plain', True), ('markdown', 'text/markdown', False)]:
            (data / 'applications' / f'{name}.desktop').write_text(
                f'[Desktop Entry]\nType=Application\nName=Private {name} fixture\n'
                f'Exec={handler} {name} %f\nTerminal={str(use_terminal).lower()}\nMimeType={mime};\n')
        (config / 'mimeapps.list').write_text('[Default Applications]\ntext/plain=text.desktop\ntext/markdown=markdown.desktop\n')
        shutil.copy(source, ui / source.name)
        (ui / 'shell.qml').write_text('''import QtQuick
import Quickshell
ShellRoot {
    property var result: null
    DesktopFileOpener {
        id: opener
        onOpened: result = {ok: true, opening: opening}
        onFailed: message => result = {ok: false, error: message, opening: opening}
        Component.onCompleted: open(Quickshell.env("OPEN_URL"))
    }
    Timer {
        interval: 100
        repeat: true
        running: true
        onTriggered: if (result !== null) { console.log("OPEN_RESULT " + JSON.stringify(result)); Qt.quit(); }
    }
}''')
        environment = dict(os.environ, XDG_CONFIG_HOME=str(config), XDG_DATA_HOME=str(data),
                           PATH=str(commands) + ':' + os.environ['PATH'], OPEN_RECORD=str(record),
                           QT_QPA_PLATFORM='offscreen', QT_QPA_PLATFORMTHEME='generic',
                           QT_QUICK_BACKEND='software', GIO_USE_VFS='local')
        # Avoid activation of applications on the user's session bus.
        environment.pop('DBUS_SESSION_BUS_ADDRESS', None)
        environment.pop('DISPLAY', None)
        environment.pop('WAYLAND_DISPLAY', None)

        def run(url, env=None):
            result = subprocess.run(['quickshell', '--path', str(ui), '--no-duplicate', '--no-color'],
                                    env=dict(environment if env is None else env, OPEN_URL=url),
                                    text=True, capture_output=True, timeout=20)
            output = result.stdout + result.stderr
            assert result.returncode == 0, output
            lines = [line.split('OPEN_RESULT ', 1)[1] for line in output.splitlines() if 'OPEN_RESULT ' in line]
            assert len(lines) == 1, output
            state = json.loads(lines[0])
            assert not state['opening'], state
            return state

        # libmagic sees Python source in .txt and plain text in .md. GIO must
        # preserve the file-type associations and launch Terminal=true correctly.
        for name, text, kind, use_terminal in [
                ('python content ü space.txt', '#!/usr/bin/python3\nprint("test")\n', 'text', True),
                ('notes ü space.md', '# Markdown\n\nTemporary fixture.\n', 'markdown', False),
                ('notes ü space.markdown', '# Markdown\n', 'markdown', False)]:
            path = root / name
            path.write_text(text)
            assert run(path.as_uri())['ok']
            deadline = time.monotonic() + 5
            while not record.exists():
                assert time.monotonic() < deadline, 'Desktop handler did not launch'
                time.sleep(.01)
            assert json.loads(record.read_text()) == {'args': [kind, str(path)], 'terminal': use_terminal}
            record.unlink()
        print('PASS real GIO MIME dispatch, Unicode/spaces and terminal desktop entries', flush=True)

        missing_file = run((root / 'missing.txt').as_uri())
        assert not missing_file['ok'] and 'could not open' in missing_file['error'], missing_file
        assert not run('https://example.com/')['ok']
        fake_gio = commands / 'gio'
        fake_gio.write_text('#!/bin/sh\necho "Private handler launch failure" >&2\nexit 17\n')
        fake_gio.chmod(0o755)
        failed = run(path.as_uri())
        assert not failed['ok'] and 'Private handler launch failure' in failed['error'] and '17' in failed['error'], failed
        # A minimal PATH keeps Quickshell visible but makes gio unavailable.
        missing_commands = root / 'missing-commands'; missing_commands.mkdir()
        (missing_commands / 'quickshell').symlink_to(shutil.which('quickshell'))
        failed = run(path.as_uri(), dict(environment, PATH=str(missing_commands)))
        assert not failed['ok'] and 'Install GLib' in failed['error'], failed
        fake_gio.write_text('#!/bin/sh\nexec /usr/bin/sleep 30\n')
        failed = run(path.as_uri())
        assert not failed['ok'] and '15 seconds' in failed['error'], failed
        print('PASS missing file/opener, remote URL rejection, launch stderr and timeout', flush=True)


if __name__ == '__main__':
    main()
