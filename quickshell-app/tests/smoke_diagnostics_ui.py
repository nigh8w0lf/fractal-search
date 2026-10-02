#!/usr/bin/env python3
"""Exercise model-independent diagnostic UI in an offscreen Quickshell process."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    source = Path(__file__).resolve().parents[1] / 'ui' / 'RuntimeDiagnostics.qml'
    report = {
        'schemaVersion': 1, 'selectedProvider': 'CUDA', 'selectedDeviceId': 0,
        'cudaEnabledBuild': True, 'platform': 'linux', 'summary': 'Fixture runtime check.',
        'checksPassed': False, 'nvidiaDriver': {'state': 'available', 'detail': 'Fixture driver'},
        'gpus': [{'index': 0, 'name': 'Fixture GPU', 'driverVersion': '580'}],
        'cudaToolkit': {'state': 'optional', 'detail': 'nvcc is not required'},
        'linkerCache': {'state': 'available', 'detail': 'Fixture cache'},
        'providerDependencies': {'state': 'missing', 'detail': 'libcudart.so.12 not found'},
        'libraries': [{'name': 'libcudart.so.12', 'component': 'CUDA runtime',
                       'state': 'notVisible', 'paths': ['/tmp/cuda/libcudart.so.12']}],
        'probeErrors': [], 'installSteps': ['Fixture setup advice'],
        'librarySearchLocations': ['/tmp/cuda'], 'ldLibraryPath': '/tmp/diagnostic-runtime',
        'cudaVisibleDevices': None,
    }
    with tempfile.TemporaryDirectory(prefix='fractal-diagnostics-ui-') as directory:
        root = Path(directory)
        shutil.copy(source, root / source.name)
        fixture = root / 'fixture.py'
        fixture.write_text('''#!/usr/bin/env python3
import json, os, sys, time
assert os.environ['LD_LIBRARY_PATH'] == '/tmp/diagnostic-runtime'
mode = os.environ['DIAGNOSTIC_CASE']
if mode == 'nonzero':
    print('Fixture missing runtime error', file=sys.stderr)
    sys.exit(17)
if mode == 'malformed':
    print('{invalid JSON')
elif mode == 'oversized':
    print('x' * 65537)
elif mode == 'timeout':
    time.sleep(25)
else:
    print(os.environ['DIAGNOSTIC_REPORT'])
''')
        fixture.chmod(0o755)
        (root / 'shell.qml').write_text('''import QtQuick
import Quickshell
ShellRoot {
    RuntimeDiagnostics {
        id: diagnostic
        executable: Quickshell.env("DIAGNOSTIC_EXECUTABLE")
        provider: "cuda"
        deviceId: 0
        libraryDirectory: "/tmp/diagnostic-runtime"
        Component.onCompleted: refresh()
    }
    Timer {
        interval: 100
        repeat: true
        running: true
        onTriggered: {
            if (!diagnostic.checking) {
                console.log("DIAGNOSTIC_RESULT " + JSON.stringify({error: diagnostic.errorMessage, details: diagnostic.details, summary: diagnostic.summary}));
                Qt.quit();
            }
        }
    }
}''')
        base = os.environ.copy()
        base.update(QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software',
                    QT_QUICK_CONTROLS_STYLE='Basic', QT_QPA_PLATFORMTHEME='generic',
                    DIAGNOSTIC_REPORT=json.dumps(report))
        base.pop('WAYLAND_DISPLAY', None)
        for case in ('success', 'nonzero', 'malformed', 'oversized', 'missing', 'timeout'):
            env = dict(base, DIAGNOSTIC_CASE=case,
                       DIAGNOSTIC_EXECUTABLE=str(root / 'missing') if case == 'missing' else str(fixture))
            result = subprocess.run(['quickshell', '--path', str(root), '--no-duplicate', '--no-color'],
                                    env=env, capture_output=True, text=True, timeout=24)
            output = result.stdout + result.stderr
            assert result.returncode == 0, output
            lines = [line.split('DIAGNOSTIC_RESULT ', 1)[1] for line in output.splitlines() if 'DIAGNOSTIC_RESULT ' in line]
            assert len(lines) == 1, output
            state = json.loads(lines[0])
            if case == 'success':
                assert not state['error'], state
                assert 'INSTALLED, outside loader path' in state['details'], state
                assert 'nvcc is not required' in state['details'], state
                assert '/tmp/diagnostic-runtime' in state['details'], state
            else:
                assert state['error'], state
                if case == 'nonzero':
                    assert 'Fixture missing runtime error' in state['error'], state
                elif case == 'timeout':
                    assert 'exceeded 20 seconds' in state['error'], state
            print(f'PASS diagnostic UI {case}', flush=True)


if __name__ == '__main__':
    main()
