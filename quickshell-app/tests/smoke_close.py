#!/usr/bin/env python3
"""Close routing failures must never dispatch Close to an unrelated window."""
import json
import os
from pathlib import Path
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[2]
PACKAGING = REPO / 'quickshell-app/packaging'


def main():
    with tempfile.TemporaryDirectory(prefix='fractal-close-', dir='/tmp') as temporary:
        root = Path(temporary)
        commands = root / 'commands'
        commands.mkdir()
        log = root / 'calls'
        fake = '''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
tool=Path(sys.argv[0]).name
with open(os.environ['CLOSE_TEST_LOG'],'a') as f:f.write(json.dumps([tool,*sys.argv[1:]])+'\\n')
if tool=='hyprctl':
    if sys.argv[1]=='activewindow':print(os.environ['CLOSE_TEST_ACTIVE'])
    elif sys.argv[1]=='layers':print(os.environ['CLOSE_TEST_LAYERS'])
    else:print('ok')
elif tool=='omarchy-shell':
    print(os.environ['CLOSE_TEST_RESPONSE']);sys.exit(int(os.environ['CLOSE_TEST_IPC_EXIT']))
'''
        for name in ['hyprctl', 'omarchy-shell', 'notify-send']:
            executable = commands / name
            executable.write_text(fake)
            executable.chmod(0o755)
        base = dict(os.environ, PATH=str(commands)+':'+os.environ['PATH'],
                    CLOSE_TEST_LOG=str(log), CLOSE_TEST_ACTIVE=json.dumps({'title':'An app with "quotes"', 'address':'0x1234'}),
                    CLOSE_TEST_LAYERS='{}', CLOSE_TEST_RESPONSE='pass', CLOSE_TEST_IPC_EXIT='0')

        def route(response, success=True, closed=False, **changes):
            log.write_text('')
            result = subprocess.run([str(PACKAGING / 'close-window.sh')], env=dict(base, CLOSE_TEST_RESPONSE=response, **changes), text=True, capture_output=True)
            assert (result.returncode == 0) is success, result.stdout+result.stderr
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            dispatches = [call for call in calls if call[:2] == ['hyprctl', 'dispatch']]
            assert bool(dispatches) is closed, calls
            if closed:
                assert dispatches == [['hyprctl','dispatch','hl.dsp.window.close({window="address:0x1234"})']],dispatches
            assert sum(call[:2] == ['hyprctl','activewindow'] for call in calls) == 1
            if not success:
                assert result.stderr and any(call[0] == 'notify-send' for call in calls)

        route('handled')
        route('pass', closed=True)
        route('unknown', closed=True)
        route('unknown', success=False, CLOSE_TEST_LAYERS='{"monitor":{"levels":{"3":[{"namespace":"fractal-search"}]}}}')
        route('unknown', success=False, CLOSE_TEST_LAYERS='not json')
        route('error', success=False)
        route('pass', success=False, CLOSE_TEST_IPC_EXIT='1')
        route('pass', success=False, CLOSE_TEST_ACTIVE='{"title":"app","address":"0x1);bad"}')
        route('pass', success=False, CLOSE_TEST_ACTIVE='not json')
        route('pass', CLOSE_TEST_ACTIVE='{}')

        config = root / 'config'
        bindings = config / 'hypr/bindings.lua'
        bindings.parent.mkdir(parents=True)
        original = '-- personal configuration\no.bind("SUPER + I", "Search", "search")\n'
        bindings.write_text(original)
        plugin = config / 'plugin with spaces'

        def bind(action, success=True):
            result = subprocess.run([str(PACKAGING/'close-binding.sh'), action, str(config), str(plugin)], capture_output=True, text=True)
            assert (result.returncode == 0) is success,result.stdout+result.stderr

        bind('install')
        installed = bindings.read_text()
        assert installed.startswith(original) and 'hl.unbind("SUPER + W")' in installed
        assert installed.count('-- BEGIN Fractal Search close binding') == 1
        bind('install')
        assert bindings.read_text() == installed
        bind('remove')
        assert bindings.read_text() == original
        for override in ['o.bind("SUPER + W", "Other", "other")\n', 'hl.unbind("SUPER + W")\n', '-- BEGIN Fractal Search close binding\n']:
            bindings.write_text(original+override)
            bind('install', False)
            assert bindings.read_text() == original+override
        assert list(bindings.parent.glob('bindings.lua.bak.fractal-close.*'))
    print('Close routing: focused/unloaded/passthrough, original-window targeting, visible failures, binding preservation and idempotence passed')


if __name__ == '__main__':
    main()
