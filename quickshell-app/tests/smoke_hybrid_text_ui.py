#!/usr/bin/env python3
"""Model-free Controller JS regression checks. Does not replace native QML/render QA."""
from pathlib import Path
import re
import shutil
import subprocess


def main():
    source = Path(__file__).resolve().parents[1] / 'ui'
    controller = (source / 'Controller.qml').read_text()
    panel = (source / 'SearchPanel.qml').read_text()
    functions = re.findall(r'^    function (\w+)\((.*?)\) \{\n(.*?)^    \}\n', controller, re.M | re.S)
    script = '\n'.join(f'function {name}({args}) {{\n{body}\n}}' for name, args, body in functions)
    assert shutil.which('node'), 'Node is required for the model-free Controller JS regression test'
    # Exercise the actual JS functions extracted from the shipped Controller.
    harness = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
const methods = __METHODS__;
function fixture(id, mediaKind, similarity = null, hybridScore = null) {
    return {assetId:id, filename:id+'.txt', path:'/synthetic/'+id, mediaKind, segmentIndex:0,
        startMs:null, endMs:null, textExcerpt:mediaKind === 'text' ? 'bounded excerpt' : null,
        thumbnailUrl:['image','video'].includes(mediaKind) ? 'file:///cached/'+id+'.webp' : '',
        width:mediaKind==='image' ? 96 : 0, height:mediaKind==='image' ? 64 : 0, similarity, hybridScore};
}
function create() {
    const rows = [];
    const timer = () => ({restart(){}, stop(){}, start(){}});
    const commands = [];
    const c = {console:{error(){}}, ready:true, starting:false, closing:false, indexing:false,
        installingModels:false, loading:false, searchStarting:false, searchWarming:false,
        stoppedCleanly:false, restarting:false, deferredSearch:false, pendingSearchText:'',
        query:'', activeQuery:'', sourceId:null, mediaKind:null, queryGeneration:0,
        resultRequestSerial:0, resultRequestPending:false, previewGeneration:0, previewLoading:false, previewRequestPending:false, previewRequestSerial:0,
        previewUrl:'', previewError:'', selectedAsset:null, selectedIndex:-1, total:0, page:'summon',
        pending:{}, nextId:1, discardedBeforeId:0, errorMessage:'', notice:'', status:{settings:{topMatches:6}, catalog:{sources:[]}},
        settings:{topMatches:6}, previewTimer:timer(), connectTimer:timer(), manageService:false,
        serviceFailed(){}, stopped(){}, commands,
        results:{get count(){return rows.length}, append(row){rows.push(row)}, clear(){rows.length=0}, get(i){return rows[i]}},
        worker:{connected:true, write(line){commands.push(JSON.parse(line))}, flush(){}}};
    c.app = c;
    vm.createContext(c);
    vm.runInContext(methods, c);
    c.refreshStatus = () => {};
    c.reply = (req, items, error='') => c.receive(JSON.stringify({id:req.id, ok:!error, error, result:{items,total:items.length}}));
    return c;
}
const all = [fixture('image','image',.95), fixture('text','text',.8), fixture('audio','audio',.4)];
let c = create();
c.browse(false); let old = c.commands.shift();
assert.equal(old.command.type,'browse'); assert.equal(old.command.mediaKind,null);
c.reply(old,all);
assert.equal(c.results.count,3); assert.equal(c.selectedAsset.assetId,'image');
assert.equal(c.scoreLabel(c.results.get(0)),'Cosine 0.950');
c.setMediaKind('text'); old = c.commands.shift();
assert.equal(old.command.type,'browse'); assert.equal(old.command.mediaKind,'text');
c.reply(old,[fixture('text','text')]);
assert.equal(c.previewText(),'bounded excerpt'); assert.equal(c.previewUrl,'');
assert.equal(c.scoreLabel(c.results.get(0)),'');
c.search('exact identifier'); old = c.commands.shift();
assert.equal(old.command.type,'search'); assert.equal(old.command.mediaKind,'text');
c.reply(old,[fixture('lexical','text',.2,.032),fixture('semantic','text',.9,.016)]);
assert.equal(c.results.get(0).assetId,'lexical','Never re-sort fused results by cosine');
assert.equal(c.scoreLabel(c.results.get(0)),'Hybrid score 0.03200');
assert.equal(c.selectedAsset.hybridScore,.032);

// New intents coalesce to the latest paired folder/type/query scope without stale results/errors/progress.
c.search('older'); old = c.commands.shift();
c.search('latest'); c.setSourceId(17); c.setMediaKind('video');
assert.equal(c.commands.length,0); assert.equal(c.results.count,0);
c.receive(JSON.stringify({id:old.id,event:'searchProgress',payload:{stage:'starting'}}));
assert.equal(c.searchStarting,false,'Stale warmup progress must not change current loading state');
c.reply(old,[], 'stale error');
assert.equal(c.errorMessage,''); assert.equal(c.results.count,0);
let current = c.commands.shift();
assert.equal(current.command.query,'latest'); assert.equal(current.command.sourceId,17); assert.equal(current.command.mediaKind,'video');
c.reply(current,[fixture('video','video',.6)]);
assert.equal(c.selectedAsset.assetId,'video'); assert.equal(c.previewUrl,'file:///cached/video.webp');
assert.equal(c.loading,false);

// A query edit invalidates the old response before the debounce timer fires.
c.search('before debounce'); old = c.commands.shift();
c.editQuery('after debounce'); c.reply(old,[fixture('wrong','video',.99)]);
assert.equal(c.results.count,0); assert.equal(c.loading,false);
c.search(c.query); current = c.commands.shift(); assert.equal(current.command.query,'after debounce');
c.reply(current,[fixture('right','video',.8)]); assert.equal(c.selectedAsset.assetId,'right');

// Re-selecting the same filter submits the current edit when the UI stops debounce.
c.editQuery('same filter latest'); c.setMediaKind('video'); current=c.commands.shift();
assert.equal(current.command.query,'same filter latest'); c.reply(current,[fixture('same-filter','video',.6)]);

// Scope checking is independent of generation checking.
c.search('paired scope'); old = c.commands.shift(); c.sourceId = 21;
c.reply(old,[fixture('wrong-folder','video',.99)]); assert.equal(c.results.count,0);
c.browse(false); current = c.commands.shift(); c.reply(current,[fixture('correct-folder','video',.7)]);

// Retain latest intent across service startup/warmup, including changes before ready.
c = create(); c.ready=false; c.worker.connected=false; c.starting=true;
c.search('warmup first'); c.setMediaKind('text'); c.setSourceId(3); c.search('warmup latest');
assert.equal(c.commands.length,0);
c.worker.connected=true;
c.receive(JSON.stringify({event:'ready',payload:{protocolVersion:1,textModelWarming:true,settings:{topMatches:6},catalog:{sources:[]}}}));
current=c.commands.shift(); assert.equal(current.command.query,'warmup latest'); assert.equal(current.command.mediaKind,'text'); assert.equal(current.command.sourceId,3);
c.receive(JSON.stringify({event:'warmupFinished',payload:{ok:true}}));
c.reply(current,[fixture('warmed','text',.7,.025)]); assert.equal(c.selectedAsset.assetId,'warmed');

// A dead connection callback cannot consume the newer connection's pending state.
c.search('old connection'); old=c.commands.shift(); const oldCallback=c.pending[old.id].callback;
c.resetConnectionRequests(); c.ready=false;
c.editQuery('new connection'); c.search(c.query);
c.receive(JSON.stringify({event:'ready',payload:{protocolVersion:1,settings:{topMatches:6},catalog:{sources:[]}}}));
current=c.commands.shift(); oldCallback({items:[fixture('stale','text',1,.04)],total:1});
c.reply(old,[], 'stale socket failure');
c.receive(JSON.stringify({id:old.id,event:'searchProgress',payload:{stage:'starting'}}));
assert.equal(c.errorMessage,''); assert.equal(c.searchStarting,false);

assert.equal(c.resultRequestPending,true); assert.equal(c.results.count,0);
c.reply(current,[fixture('reconnected','text',.3,.02)]); assert.equal(c.selectedAsset.assetId,'reconnected');

// Preview responses are guarded by both query and selection generations, including repeated IDs.
c = create(); c.browse(false); current=c.commands.shift(); c.reply(current,all);
c.loadPreview(); old=c.commands.shift(); assert.equal(old.command.type,'preview');
c.setMediaKind('text'); current=c.commands.shift(); c.reply(current,[fixture('text','text')]);
c.reply(old,[], 'stale preview error'); assert.equal(c.errorMessage,''); assert.equal(c.previewError,''); assert.equal(c.previewUrl,''); assert.equal(c.previewLoading,false);
c.loadPreview(); assert.equal(c.commands.length,0,'Text must never request an image preview');
c.setMediaKind('image'); current=c.commands.shift(); c.reply(current,[fixture('image','image')]);
c.loadPreview(); old=c.commands.shift(); c.select(0); c.loadPreview();
assert.equal(c.commands.length,0,'Keep only one outstanding preview request');
c.receive(JSON.stringify({id:old.id,ok:true,result:{url:'file:///stale.png'}}));
assert.equal(c.previewLoading,true); assert.equal(c.previewUrl,'');
c.loadPreview(); current=c.commands.shift();
c.receive(JSON.stringify({id:current.id,ok:true,result:{url:'file:///fresh.png'}}));
assert.equal(c.previewLoading,false); assert.equal(c.previewUrl,'file:///fresh.png');
// Appending a page keeps the current selection/preview generation alive.
c.browse(true); const page=c.commands.shift(); assert.equal(page.command.offset,1);
c.reply(page,[fixture('image2','image')]); assert.equal(c.results.count,2); assert.equal(c.selectedAsset.assetId,'image');
// Closing during a search invalidates retained text and late results.
c.dismissRequested=()=>{}; c.search('closing'); old=c.commands.shift(); c.dismiss();
c.reply(old,[fixture('closed','image',.9)]); assert.equal(c.results.count,0); assert.equal(c.activeQuery,'');
console.log('PASS: actual Controller JS media/query/folder scope, bounded coalescing, model-free browsing, hybrid order/labels, debounce, warmup/reconnect and stale previews');
'''
    import json
    harness = harness.replace('__METHODS__', json.dumps(script))
    subprocess.run(['node', '-e', harness], check=True, timeout=20)
    for label in ['All media', 'Images', 'Text', 'Audio', 'Video']:
        assert f'label: "{label}"' in panel
    assert 'objectName: "mediaKindFilter"' in panel
    assert 'app.setMediaKind(currentValue)' in panel and 'app.setSourceId(currentValue)' in panel
    assert 'app.editQuery(text)' in panel and 'app.scoreLabel(resultRow.model)' in panel
    assert 'Layout.minimumWidth: 100' in panel
    print('PASS: static selector, score labels and narrow-width constraints; native QML/render QA remains separate')


if __name__ == '__main__':
    main()
