#!/usr/bin/env python3
"""Model-free QMD UI contracts and shipped diagnostics JS; not native/render QA."""
from pathlib import Path
import json
import re
import shutil
import subprocess


def main():
    source = Path(__file__).resolve().parents[1] / 'ui'
    settings = (source / 'SettingsPage.qml').read_text()
    panel = (source / 'SearchPanel.qml').read_text()
    grouped = (source / 'GroupedTextMatches.qml').read_text()
    policy = (source / 'ModelIdlePolicy.qml').read_text()
    runtime = (source / 'ModelRuntimeSection.qml').read_text()
    accelerator = (source / 'AcceleratorSection.qml').read_text()
    controller = (source / 'Controller.qml').read_text()
    directory = (source / 'qmldir').read_text()
    for name in ['GroupedTextMatches', 'ModelIdlePolicy', 'ModelRuntimeSection']:
        assert f'{name} 1.0 {name}.qml' in directory
    assert 'objectName: "groupTextByFileToggle"' in settings
    assert 'groupTextByFile: checked' in settings
    assert 'All media searches and browsing keep segment results.' in settings
    assert 'onClicked: page.app.reindexAll()' in settings, 'Ordinary rescan stays ordinary'
    assert 'onClicked: page.rebuildText()' in settings
    assert 'id: rebuildTextDialog\n                parent: canvas' in panel
    assert 'onAccepted: app.reindexAll(true)' in panel
    assert 'Original files stay in place.' in panel
    confirmation = (source / 'SearchConfirmationDialog.qml').read_text()
    assert 'onOpened: cancelButton.forceActiveFocus()' in confirmation
    assert 'if (rebuildTextDialog.visible)\n                rebuildTextDialog.close();' in panel
    assert '!restartDialog.visible && !rebuildTextDialog.visible' in panel
    assert panel.count('GroupedTextMatches {') == 2, 'Preview and list-only/narrow layouts both expose matches'
    assert panel.count('ResultActions {') == 2, 'Keep whole-file actions in both layouts'
    assert 'onAssetIdChanged: expanded = false' in grouped
    assert 'Math.min(section.maximumExcerptHeight, excerpts.implicitHeight)' in grouped
    assert 'ScrollBar.horizontal.policy: ScrollBar.AlwaysOff' in grouped
    assert 'textFormat: TextEdit.PlainText' in grouped and 'readOnly: true' in grouped
    assert 'property bool expanded: false' in grouped and 'property bool expanded: false' in runtime
    assert 'from: 1' in policy and 'to: 86400' in policy
    assert '"Release when idle", "Keep warm"' in policy
    assert 'currentIndex === 1 ? 0' in policy
    assert 'settingKey: "textIdleSeconds"' in runtime and 'defaultSeconds: 900' in runtime
    assert 'settingKey: "mediaIdleSeconds"' in runtime and 'defaultSeconds: 300' in runtime
    assert 'Service snapshot, refreshed every 5 seconds' in runtime
    assert 'panelVisible: window.visible' in panel
    assert 'performanceVisible: panelVisible && visible && currentTab === 2' in settings
    assert 'statusUpdatesEnabled: page.performanceVisible' in settings
    assert 'interval: 5000' in accelerator and 'triggeredOnStart: true' in accelerator
    assert 'section.diagnostics.refresh();\n                    if (section.app.ready && !section.app.busy)\n                        section.app.refreshStatus();' in accelerator
    assert 'visible: app.results.count < app.total\n' in panel, 'Bounded search pages must expose Load more too'
    assert 'does not measure RAM or GPU memory' in runtime
    for key in ['idleTimeoutSeconds', 'idleForMs', 'loadCount', 'releaseCount', 'lastLoadDurationMs', 'lastReloadDurationMs']:
        assert key in runtime
    functions = re.findall(r'^    function (\w+)\((.*?)\) \{\n(.*?)^    \}\n', runtime, re.M | re.S)
    methods = '\n'.join(f'function {name}({args}) {{\n{body}\n}}' for name, args, body in functions)
    model_ready = re.search(r'    readonly property bool modelReady: (.*?)\n    readonly property bool gpuActive:', accelerator, re.S).group(1)
    polling = re.search(r'^        running: (.*)$', accelerator, re.M).group(1)
    refresh = re.search(r'^    function refreshStatus\(\) \{\n.*?^    \}\n', controller, re.M | re.S).group(0)
    assert shutil.which('node'), 'Node is required for model-free diagnostics JS checks'
    harness = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
__METHODS__
assert.equal(duration(null),'not recorded');
assert.equal(duration(undefined),'not recorded');
assert.equal(duration(0),'0 ms');
assert.equal(duration(42.4),'42 ms');
assert.equal(duration(2500),'2.50 s');
assert.equal(summary('Text',null),'Text: unavailable');
assert.equal(summary('Text',{loaded:true,idleTimeoutSeconds:null}),'Text: loaded · keep warm');
assert.equal(summary('Vision',{loaded:false,idleTimeoutSeconds:300}),'Vision: released / not loaded · release after 300 s idle');
function active(ready, runtime, textModelLoaded=false) {
    const app = {ready,status:{textModelLoaded}};
    const modelSessions = runtime;
    return (__READY__);
}
assert.equal(active(true,{text:{loaded:false},vision:{loaded:true},audio:{loaded:false}}),true);
assert.equal(active(true,{text:{loaded:false},vision:{loaded:false},audio:{loaded:true}}),true);
assert.equal(active(true,{text:{loaded:false},vision:{loaded:false},audio:{loaded:false}},true),false,'Diagnostics override stale text-only status');
assert.equal(active(false,{text:{loaded:true},vision:{loaded:true},audio:{loaded:true}}),false,'Disconnected status must not imply an active provider');
assert.equal(active(true,undefined,true),true,'Older static fixtures retain text-loaded status');
assert.equal(active(true,null,false),false);
function shouldPoll(enabled, visible, ready, busy) {
    const section = {statusUpdatesEnabled:enabled, visible, app:{ready,busy}};
    return (__POLL__);
}
assert.equal(shouldPoll(true,true,true,false),true);
assert.equal(shouldPoll(false,true,true,false),false,'No polling when panel or Performance is hidden');
assert.equal(shouldPoll(true,false,true,false),false,'No polling from a hidden accelerator');
assert.equal(shouldPoll(true,true,false,false),false,'No polling while disconnected');
assert.equal(shouldPoll(true,true,true,true),false,'No polling while work is active');
const c = {ready:true, worker:{connected:true}, closing:false, pending:{}, status:null, requests:[]};
c.app=c;
c.request=(command, callback)=>{c.requests.push({command,callback});c.pending[1]={type:command.type}};
vm.createContext(c);
vm.runInContext(__REFRESH__,c);
c.refreshStatus();
c.refreshStatus();
assert.equal(c.requests.length,1,'At most one status request can remain in flight');
c.pending={};
c.requests[0].callback({runtimeDiagnostics:{text:{loaded:false}}});
assert.equal(c.status.runtimeDiagnostics.text.loaded,false,'Refreshed idle release replaces stale loaded state');
c.ready=false;c.refreshStatus();c.ready=true;
c.worker.connected=false;c.refreshStatus();c.worker.connected=true;
c.closing=true;c.refreshStatus();c.closing=false;
assert.equal(c.requests.length,1,'Disconnected and draining services do not receive refreshes');
c.pending={2:{type:'search'}};c.refreshStatus();
assert.equal(c.requests.length,2,'Other request kinds do not indefinitely suppress status');
console.log('PASS: actual diagnostics JS, nullable KeepWarm policy, timing formatting, accelerator state and bounded visible-only status refresh');
'''
    harness = harness.replace('__METHODS__', methods).replace('__READY__', model_ready).replace('__POLL__', polling).replace('__REFRESH__', json.dumps(refresh))
    subprocess.run(['node', '-e', harness], check=True, timeout=20)
    print('PASS: static grouped excerpts, bounded scroll areas, both action layouts, idle controls, persistent confirmation/default Cancel and runtime diagnostics wiring')
    print('Native Quickshell rendering, narrow-window layout and physical interactions remain untested by this script')


if __name__ == '__main__':
    main()
