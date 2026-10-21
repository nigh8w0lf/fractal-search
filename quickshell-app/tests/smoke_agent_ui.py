#!/usr/bin/env python3
"""MCP connection UI regression, with an optional private compositor.

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
    root = Path(tempfile.mkdtemp(prefix='fsa-', dir='/tmp'))
    print(f'Agent UI test artifacts: {root}', flush=True)
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
        for name in ['AgentConnectionSettings.qml', 'SearchButton.qml', 'SearchText.qml', 'SearchTheme.qml', 'qmldir']:
            shutil.copy(source / name, ui / name)
        (ui / 'Commons').symlink_to('/usr/share/omarchy/shell/Commons', target_is_directory=True)
        (ui / 'Ui').symlink_to('/usr/share/omarchy/shell/Ui', target_is_directory=True)
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
        assert 'AGENT_PASS' in output and 'FAIL!' not in output and 'Error:' not in output, output
        print('PASS: native copy actions, JSON/path escaping, guide dispatch, narrow layout and missing configuration')
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
import QtQuick.Layouts
import QtTest
import Quickshell
ShellRoot {
    QtObject {
        id: controller
        property var colors: ({foreground: "#eeeeee", background: "#111111", accent: "#88ffaa", error: "#ff8888"})
        property color secondaryText: "#bbbbbb"
        property string failure: ""
        function fail(message) { failure = message; }
    }
    FloatingWindow {
        id: window
        visible: true
        implicitWidth: 560
        implicitHeight: 550
        color: "#111111"
        AgentConnectionSettings {
            id: card
            property int testWidth: 520
            x: 20; y: 20; width: Math.min(testWidth, parent.width - 40)
            app: controller
            serverPath: "/tmp/custom config/quoted\"/fractal-search-mcp"
            property int guideRequests: 0
            onOpenGuide: guideRequests++
        }
        TextEdit { id: pasted; visible: false }
        TestCase {
            name: "Agent connection"
            when: window.visible
            function button(name) { const item = findChild(card, name); verify(item !== null); return item; }
            function checkCopy(name, expected) {
                mouseClick(button(name)); pasted.text = ""; pasted.paste();
                compare(pasted.text, expected);
            }
            function test_connection() {
                wait(500);
                checkCopy("copyAgentConfiguration", card.configuration);
                compare(JSON.parse(pasted.text).mcpServers["fractal-search"].command, card.serverPath);
                checkCopy("copyAgentPath", card.serverPath);
                mouseClick(button("agentSetupGuide")); compare(card.guideRequests, 1);
                console.log("CAPTURE_NORMAL"); wait(700);
                card.testWidth = 280; wait(500);
                compare(card.width, 280);
                for (const name of ["copyAgentConfiguration", "copyAgentPath", "agentSetupGuide"]) {
                    const item = button(name); const pos = item.mapToItem(card, 0, 0);
                    verify(pos.x + item.width <= card.width, name + " overflows");
                    verify(pos.y + item.height <= card.height, name + " clipped");
                }
                checkCopy("copyAgentConfiguration", card.configuration);
                console.log("CAPTURE_NARROW"); wait(700);
                card.serverPath = "";
                verify(!button("copyAgentConfiguration").enabled);
                verify(!button("copyAgentPath").enabled);
                verify(button("agentSetupGuide").enabled);
                card.copyValue("invalid", "path"); verify(controller.failure.includes("Reinstall"));
                console.log("CAPTURE_ERROR"); wait(700);
                console.log("AGENT_PASS");
            }
            function cleanupTestCase() { Qt.quit(); }
        }
    }
}
'''

if __name__ == '__main__':
    main()
