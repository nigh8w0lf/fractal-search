#!/usr/bin/env python3
"""First-run setup UI regression, with an optional private compositor.

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
    root = Path(tempfile.mkdtemp(prefix='fsu-', dir='/tmp'))
    print(f'Setup UI test artifacts: {root}', flush=True)
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
        ui = root / 'ui'; ui.mkdir()
        source = Path(__file__).resolve().parents[1] / 'ui'
        for name in ['SetupPanel.qml', 'SearchButton.qml', 'SearchText.qml', 'SearchTheme.qml', 'qmldir']:
            shutil.copy(source / name, ui / name)
        (ui / 'Commons').symlink_to('/usr/share/omarchy/shell/Commons', target_is_directory=True)
        (ui / 'Ui').symlink_to('/usr/share/omarchy/shell/Ui', target_is_directory=True)
        setup = ui / 'SetupPanel.qml'
        text = setup.read_text()
        start = text.index('    readonly property string helper:')
        end = text.index('    readonly property string sourceDirectory:', start)
        text = text[:start] + '    readonly property string helper: Quickshell.env("SETUP_FIXTURE")\n    property int testWidth: 520\n' + text[end:]
        text = text.replace('Math.min(Style.space(560), screen ? screen.width - 32 : 560)', 'setup.testWidth', 1)
        setup.write_text(text)
        fixture = root / 'fixture.sh'
        fixture.write_text('''#!/bin/bash
set -eu
if [[ $1 == status ]]; then
  if [[ -f "$SETUP_STATE" ]]; then cat "$SETUP_STATE"; else echo '{"stage":"idle","message":"Choose a backend to get started."}'; fi
else
  if [[ $2 == cpu ]]; then
    echo '{"stage":"error","message":"Fixture dependency missing. Install the required system package, then retry."}' > "$SETUP_STATE"
  else
    [[ $4 == "/tmp/private CUDA lib" ]] || exit 9
    echo '{"stage":"downloading","message":"Downloading verified release…"}' > "$SETUP_STATE"
    sleep 2
    echo '{"stage":"done","message":"App installed. Open Fractal Search and select Set up model."}' > "$SETUP_STATE"
  fi
fi
''')
        env['SETUP_FIXTURE'] = str(fixture)
        env['SETUP_STATE'] = str(root / 'state.json')
        (ui / 'shell.qml').write_text(QML)
        panel = spawn(['quickshell', '--path', str(ui), '--no-duplicate', '--no-color'], 'ui')
        deadline = time.monotonic() + 25
        captured = set()
        while panel.poll() is None:
            assert time.monotonic() < deadline, (root / 'ui.log').read_text()
            output = (root / 'ui.log').read_text()
            for phase in ['NORMAL', 'NARROW', 'ERROR']:
                if args.native and f'CAPTURE_{phase}' in output and phase not in captured:
                    subprocess.run(['grim', str(root / (phase.lower() + '.png'))], env=env, check=True, capture_output=True)
                    captured.add(phase)
            time.sleep(.05)
        output = (root / 'ui.log').read_text()
        assert 'SETUP_PASS' in output and 'FAIL!' not in output and 'Error:' not in output, output
        print('PASS: first-run rendering, CPU failure, CUDA retry, progress, close/reopen and narrow layout')
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
        for log in logs:
            log.close()


QML = r'''import QtQuick
import QtTest
import Quickshell
ShellRoot {
    SetupPanel { id: panel; Component.onCompleted: open("{}") }
    TestCase {
        name: "First run setup"
        when: panel.opened
        function button(name) { const item = findChild(panel, name); verify(item !== null); return item; }
        function test_setup() {
            wait(500);
            compare(panel.stage, "idle");
            verify(button("installCpu").enabled);
            console.log("CAPTURE_NORMAL"); wait(700);
            mouseClick(button("installCpu"));
            tryCompare(panel, "stage", "error", 5000);
            verify(panel.message.includes("dependency"));
            console.log("CAPTURE_ERROR"); wait(700);
            panel.testWidth = 320; wait(400);
            const cpu = button("installCpu"), cuda = button("installCuda");
            verify(cpu.width <= 272 && cuda.width <= 272);
            console.log("CAPTURE_NARROW"); wait(700);
            const directory = button("cudaLibraryDirectory");
            directory.text = "/tmp/private CUDA lib";
            mouseClick(cuda);
            tryVerify(() => panel.busy, 3000);
            verify(!cpu.enabled && !cuda.enabled && !directory.enabled);
            panel.close(); compare(panel.opened, false);
            wait(2500); panel.open("{}");
            tryCompare(panel, "stage", "done", 5000);
            console.log("SETUP_PASS");
        }
        function cleanupTestCase() { Qt.quit(); }
    }
}
'''

if __name__ == '__main__': main()
