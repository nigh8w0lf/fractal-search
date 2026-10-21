#!/usr/bin/env python3
"""Bundle install/update failure checks; all installed files and commands are isolated."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--cuda-lib-dir', type=Path, help='Runtime library directory for CUDA bundle configuration tests')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='fractal-search-installer-', dir='/tmp') as temporary:
        root = Path(temporary)
        config, data, runtime = [root / name for name in ['config', 'data', 'run']]
        options = ['--config-home', str(config), '--data-home', str(data), '--runtime-dir', str(runtime)]
        bindings = config / 'hypr/bindings.lua'
        bindings.parent.mkdir(parents=True)
        original_bindings = 'o.bind("SUPER + I", "Search", "search")\n'
        bindings.write_text(original_bindings)
        bundle = root / 'bundle'; shutil.copytree(args.bundle.resolve(), bundle)
        def install(success=True, extra=None, env=None):
            result = subprocess.run([str(bundle / 'install.sh'), *options, *(extra or [])], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert (result.returncode == 0) is success, result.stdout + result.stderr
            return result
        install(extra=['--skip-shell'])
        plugin = config / 'omarchy/plugins/fractal.search'
        unit = config / 'systemd/user/fractal-search.service'
        panel = plugin / json.loads((plugin / 'manifest.json').read_text())['entryPoints']['panel']
        assert panel.exists()
        runtime_qml = (panel.parent / 'RuntimeConfig.qml').read_text()
        assert 'pickerExecutable:' in runtime_qml
        assert 'mcpExecutable: ' + json.dumps(str(plugin / 'fractal-search-mcp')) in runtime_qml
        assert 'closeShortcutEnabled: false' in runtime_qml
        assert bindings.read_text() == original_bindings
        assert list((data / 'fractal-search/runtime').glob('*/bin/fractal-folder-picker'))
        assert list((data / 'fractal-search/runtime').glob('*/bin/fractal-search-mcp'))
        installed_license = list((data / 'fractal-search/runtime').glob('*/LICENSE'))
        assert len(installed_license) == 1
        assert installed_license[0].read_bytes() == (bundle / 'LICENSE').read_bytes()
        assert (installed_license[0].parent / 'LANCEDB-LICENSE').read_bytes() == (bundle / 'LANCEDB-LICENSE').read_bytes()
        installed_runtime = installed_license[0].parent
        for name in ['AGENTS.md', 'AGENT-SETUP.md', 'NOTICE']:
            assert (installed_runtime / name).read_bytes() == (bundle / name).read_bytes()
        for source in (bundle / 'third-party').rglob('*'):
            if source.is_file():
                assert (installed_runtime / source.relative_to(bundle)).read_bytes() == source.read_bytes()
        assert list((installed_runtime / 'third-party/sources').glob('option-ext-*.tar.gz'))
        assert list((installed_runtime / 'third-party/sources').glob('eigen-*.zip'))
        mcp = plugin / 'fractal-search-mcp'
        help_result = subprocess.run([str(mcp), '--help'], text=True, capture_output=True)
        assert help_result.returncode == 0, help_result.stdout + help_result.stderr
        assert '--data-dir' in help_result.stdout
        runtime_config = plugin / '.fractal-search-runtime.json'
        assert json.loads(runtime_config.read_text()) == {'provider': 'cpu', 'deviceId': None, 'cudaLibraryDirectory': None}
        assert 'backendExecutable:' in runtime_qml and 'configuredProvider: "cpu"' in runtime_qml
        assert not list(plugin.rglob('shell.qml'))
        assert json.loads((plugin / 'manifest.json').read_text())['id'] == 'fractal.search'
        assert 'daemon --warm-search --data-dir' in unit.read_text()
        assert 'WantedBy=graphical-session.target' in unit.read_text()
        assert 'PartOf=graphical-session.target' in unit.read_text()
        enabled_link = config / 'systemd/user/graphical-session.target.wants/fractal-search.service'
        enabled_link.parent.mkdir(parents=True)
        enabled_link.symlink_to(unit)
        assert 'KillMode=mixed' in unit.read_text()
        catalog = data / 'fractal-search/catalog.lancedb/assets.lance/marker'
        catalog.parent.mkdir(parents=True)
        catalog.write_bytes(b'untouched catalog marker')
        install(extra=['--skip-shell'])
        assert catalog.read_bytes() == b'untouched catalog marker'
        assert enabled_link.is_symlink() and enabled_link.resolve() == unit
        original_ui = panel.read_bytes()
        original_unit = unit.read_bytes()
        assert '--provider cpu' in original_unit.decode()
        assert 'Provider must be cpu or cuda' in install(False, ['--skip-shell', '--provider', 'auto']).stderr
        assert 'require --provider cuda' in install(False, ['--skip-shell', '--device-id', '1']).stderr
        assert unit.read_bytes() == original_unit
        saved_config = runtime_config.read_bytes()
        runtime_config.write_text('{"provider":"cpu","unexpected":true}')
        assert 'Invalid installed inference configuration' in install(False, ['--skip-shell']).stderr
        assert unit.read_bytes() == original_unit
        runtime_config.write_bytes(saved_config)
        if json.loads((bundle / 'bundle.json').read_text())['cuda']:
            cuda_options = ['--skip-shell', '--provider', 'cuda', '--device-id', '0']
            if args.cuda_lib_dir:
                cuda_options += ['--cuda-lib-dir', str(args.cuda_lib_dir.resolve())]
            install(extra=cuda_options)
            assert '--provider cuda --device-id 0' in unit.read_text()
            cuda_config = runtime_config.read_bytes()
            cuda_unit = unit.read_bytes()
            cuda_destination = (plugin / '.fractal-search-install').read_text().strip()
            expected_loader = cuda_destination + '/bin' + (':' + str(args.cuda_lib_dir.resolve()) if args.cuda_lib_dir else '')
            assert f'Environment="LD_LIBRARY_PATH={expected_loader}"' in unit.read_text()
            cuda_panel = plugin / json.loads((plugin / 'manifest.json').read_text())['entryPoints']['panel']
            assert 'runtimeLibraryPath: ' + json.dumps(expected_loader) in (cuda_panel.parent / 'RuntimeConfig.qml').read_text()
            install(extra=['--skip-shell'])
            assert runtime_config.read_bytes() == cuda_config
            assert unit.read_bytes() == cuda_unit
            if args.cuda_lib_dir:
                assert 'Environment="LD_LIBRARY_PATH=' in unit.read_text()
            assert 'CUDA device ID' in install(False, ['--skip-shell', '--provider', 'cuda', '--device-id', '-1']).stderr
            assert unit.read_bytes() == cuda_unit
            install(extra=['--skip-shell', '--provider', 'cpu'])
            assert unit.read_bytes() == original_unit
        else:
            assert 'requires a CUDA bundle' in install(False, ['--skip-shell', '--provider', 'cuda']).stderr
            assert unit.read_bytes() == original_unit
        qml = bundle / 'ui/SearchPanel.qml'
        valid = qml.read_bytes(); qml.write_bytes(valid + b'corruption')
        damaged = install(False, ['--skip-shell'])
        assert 'FAILED' in damaged.stdout + damaged.stderr
        assert panel.read_bytes() == original_ui
        qml.write_bytes(valid)
        legal = bundle / 'third-party/onnxruntime-1.24.2/ThirdPartyNotices.txt'
        valid_legal = legal.read_bytes()
        legal.write_bytes(valid_legal + b'corruption')
        damaged = install(False, ['--skip-shell'])
        assert 'FAILED' in damaged.stdout + damaged.stderr
        assert (installed_runtime / legal.relative_to(bundle)).read_bytes() == valid_legal
        assert panel.read_bytes() == original_ui
        legal.write_bytes(valid_legal)
        marker = plugin / '.fractal-search-install'; saved = marker.read_bytes(); marker.unlink()
        assert 'unmanaged' in install(False, ['--skip-shell']).stderr
        marker.write_bytes(saved)
        fakebin = root / 'commands'; fakebin.mkdir()
        wrappers = {
            'systemctl': """#!/bin/sh
case "$2" in
  show) cat "$SEARCH_TEST_STATE" ;;
  stop) echo inactive > "$SEARCH_TEST_STATE" ;;
  start)
    if [ -e "$SEARCH_TEST_FAIL_START" ]; then rm "$SEARCH_TEST_FAIL_START"; echo 'injected start failure' >&2; exit 9; fi
    echo active > "$SEARCH_TEST_STATE" ;;
  disable) rm "$XDG_CONFIG_HOME/systemd/user/graphical-session.target.wants/fractal-search.service" ;;
esac
exit 0
""",
            'omarchy-shell': """#!/bin/sh
if [ "$2" = call ]; then basename "$(cat "$SEARCH_TEST_PLUGIN/.fractal-search-install")"; else echo ok; fi
""",
            'omarchy': '#!/bin/sh\nif [ "$2" = enable ] && [ "$SEARCH_TEST_FAIL_ENABLE" = 1 ]; then echo "injected enable failure" >&2; exit 9; fi\nexit 0\n',
            'hyprctl': '#!/bin/sh\nexit 0\n',
        }
        for name, text in wrappers.items():
            p = fakebin / name; p.write_text(text); p.chmod(0o755)
        service_state = root / 'service-state'; service_state.write_text('active\n')
        fail_start = root / 'fail-start'
        environment = dict(os.environ, PATH=str(fakebin)+':'+os.environ['PATH'],
                           SEARCH_TEST_STATE=str(service_state), SEARCH_TEST_PLUGIN=str(plugin),
                           SEARCH_TEST_FAIL_START=str(fail_start), SEARCH_TEST_FAIL_ENABLE='1')
        obsolete = data / 'fractal-search/runtime/0.1.0-aaaaaaaaaaaa-cpu'
        if not obsolete.exists():
            (obsolete / 'bin').mkdir(parents=True)
            (obsolete / 'bin/fractal-search').write_text('old build fixture')
            (obsolete / 'bundle.json').write_text('{}')
        failed = install(False, env=environment)
        assert obsolete.exists(), 'Failed installation pruned an older build'
        assert 'restoring the previous' in failed.stderr
        assert service_state.read_text().strip() == 'active', 'Rollback did not restore the active service'
        assert panel.read_bytes() == original_ui
        assert unit.read_bytes() == original_unit
        assert catalog.read_bytes() == b'untouched catalog marker'
        success_environment = dict(environment, SEARCH_TEST_FAIL_ENABLE='0')
        install(env=success_environment)
        assert service_state.read_text().strip() == 'active', 'Update did not restart the service'
        service_state.write_text('inactive\n')
        install(env=success_environment)
        assert service_state.read_text().strip() == 'inactive', 'Update started a previously inactive service'
        service_state.write_text('active\n')
        fail_start.touch()
        failed_start = install(False, env=success_environment)
        assert 'injected start failure' in failed_start.stderr
        assert service_state.read_text().strip() == 'active', 'Start failure rollback did not restore service'
        assert unit.read_bytes() == original_unit
        assert panel.read_bytes() == original_ui
        ldd = fakebin / 'ldd'; ldd.write_text('#!/bin/sh\necho "libheif.so.1 => not found"\n'); ldd.chmod(0o755)
        assert 'native runtime dependencies' in install(False, ['--skip-shell'], environment).stderr
        assert unit.read_bytes() == original_unit
        install(extra=['--skip-shell', '--bind-close'])
        assert not obsolete.exists(), 'Successful installation did not prune an older build'
        installed_bindings = bindings.read_text()
        assert installed_bindings.startswith(original_bindings)
        assert installed_bindings.count('-- BEGIN Fractal Search close binding') == 1
        selected = plugin / json.loads((plugin / 'manifest.json').read_text())['entryPoints']['panel']
        assert 'closeShortcutEnabled: true' in (selected.parent / 'RuntimeConfig.qml').read_text()
        assert (plugin / 'close-window.sh').is_file()
        install(extra=['--skip-shell'])
        assert bindings.read_text() == installed_bindings
        assert 'closeShortcutEnabled: true' in (selected.parent / 'RuntimeConfig.qml').read_text()
        ldd.unlink()
        failed = install(False, env=environment)
        assert 'restoring the previous' in failed.stderr
        assert service_state.read_text().strip() == 'active', 'Rollback did not restore the active service'
        assert bindings.read_text() == installed_bindings
        uninstall = subprocess.run([str(bundle / 'uninstall.sh')], env=dict(environment, XDG_CONFIG_HOME=str(config)), text=True, capture_output=True)
        assert uninstall.returncode == 0,uninstall.stdout+uninstall.stderr
        assert bindings.read_text() == original_bindings
        assert catalog.read_bytes() == b'untouched catalog marker'
        assert not plugin.exists()
        assert not enabled_link.is_symlink()
    print('Installer: install/update, stable MCP command, explicit provider/persistence, catalog preservation, checksum failure, unmanaged paths, missing runtime, close binding update/rollback, active/inactive service preservation, failed-start recovery and uninstall passed')


if __name__ == '__main__':
    main()
