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
        get busy(){return this.loading || this.indexing || this.installingModels || this.closing || this.restarting},
        reindexQueue:[], reindexTextPassages:false,
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
// Grouped Text results preserve winning order and bounded additional passage metadata.
c = create(); c.mediaKind='text'; c.sourceId=31; c.search('grouped query'); current=c.commands.shift();
const winner = {...fixture('winner','text',.2,.035), path:'/synthetic/notes.md', startByte:128, endByte:512, startLine:7, endLine:18};
const extra = {...fixture('extra','text',.9,.028), path:winner.path, segmentIndex:3, startByte:768, endByte:900, startLine:30, endLine:30};
c.reply(current,[{...winner,additionalMatches:[extra]}]);
assert.equal(c.results.count,1); assert.equal(c.selectedAsset.assetId,'winner');
assert.equal(c.selectedAsset.additionalMatchCount,1);
assert.equal(c.additionalMatches(c.selectedAsset)[0].assetId,'extra');
assert.equal(c.additionalMatches(c.selectedAsset)[0].hasHybridScore,true);
assert.equal(c.mediaLabel(c.selectedAsset),'Text · passage 1 · lines 7–18');
assert.equal(c.mediaLabel(c.additionalMatches(c.selectedAsset)[0]),'Text · passage 4 · line 30');
assert.equal(c.mediaLabel(c.resultAsset({...winner,startLine:null,endLine:null},true)),'Text · passage 1 · bytes 128–512');
// A byte-bounded grouped page can be shorter than topMatches; append from its
// actual item count without replacing selection or dropping extra passages.
c.total=2;c.browse(true);current=c.commands.shift();
assert.equal(current.command.type,'search');assert.equal(current.command.offset,1);
c.reply(current,[{...winner,assetId:'other-file',path:'/synthetic/other.md'}]);
assert.equal(c.results.count,2);assert.equal(c.selectedAsset.assetId,'winner');
assert.equal(c.additionalMatches(c.selectedAsset)[0].assetId,'extra');
c.revealOriginal(); current=c.commands.shift(); assert.equal(current.command.assetId,'winner');
c.receive(JSON.stringify({id:current.id,ok:false,error:'fixture reveal failure'}));
assert.equal(c.previewText(),'bounded excerpt');
assert.equal(c.resultAsset({...winner,additionalMatches:[{...extra,path:'/other/file'}]},true),null);
assert.equal(c.resultAsset({...winner,additionalMatches:[extra,extra]},true),null);
assert.equal(c.resultAsset({...winner,additionalMatches:[{...extra,additionalMatches:[winner]}]},true),null);
assert.equal(c.resultAsset({...winner,additionalMatches:Array(6).fill(extra)},true),null);
assert.equal(c.resultAsset({...winner,additionalMatches:false},true),null);
assert.equal(c.resultAsset({...winner,textExcerpt:'x'.repeat(2049)},true),null);
assert.equal(c.resultAsset({...winner,textExcerpt:'漢'.repeat(683)},true),null,'Excerpt cap is UTF-8 bytes');
assert.equal(c.resultAsset({...winner,textExcerpt:'😀'.repeat(513)},true),null);
assert.ok(c.resultAsset({...winner,textExcerpt:'😀'.repeat(512)},true));
assert.equal(c.resultAsset({...winner,textExcerpt:'\uD800'},true),null,'Reject unpaired UTF-16 surrogate');
assert.equal(c.resultAsset({...winner,additionalMatches:Array.from({length:5},(_,i)=>({...extra,assetId:'extra'+i}))},true).additionalMatchCount,5);
// Saving grouping refreshes the same query/filter scope and does not drop unrelated preferences.
c.settings={topMatches:6,groupTextByFile:false,textIdleSeconds:900,mediaIdleSeconds:300,imageBatchSize:1};
c.updateSettings({groupTextByFile:true}); current=c.commands.shift();
assert.equal(current.command.settings.groupTextByFile,true); assert.equal(current.command.settings.textIdleSeconds,900);
c.receive(JSON.stringify({id:current.id,ok:true,result:{settings:current.command.settings,catalog:{sources:[]}}}));
current=c.commands.shift(); assert.equal(current.command.query,'grouped query'); assert.equal(current.command.sourceId,31); assert.equal(current.command.mediaKind,'text');
c.reply(current,[winner]); assert.equal(c.selectedAsset.additionalMatchCount,0);
c.mediaKind=null; assert.equal(c.resultAsset({...winner,hybridScore:null,additionalMatches:[{...extra,hybridScore:null}]},true),null,'All media retains segment mode');
c.mediaKind='text'; c.activeQuery=''; assert.equal(c.resultAsset({...winner,hybridScore:null,additionalMatches:[{...extra,hybridScore:null}]},true),null,'Browsing retains segment mode');

// Idle settings use the ordinary settings update, including zero Keep warm, without a restart or search.
c=create(); c.settings={topMatches:6,groupTextByFile:false,textIdleSeconds:900,mediaIdleSeconds:300,imageBatchSize:1};
c.updateSettings({textIdleSeconds:0,mediaIdleSeconds:86400}); current=c.commands.shift();
assert.equal(current.command.settings.textIdleSeconds,0); assert.equal(current.command.settings.mediaIdleSeconds,86400);
c.receive(JSON.stringify({id:current.id,ok:true,result:{settings:current.command.settings,catalog:{sources:[]}}}));
assert.equal(c.commands.length,0); assert.equal(c.loading,false);

// Only the explicit rebuild path verifies content and forces text embedding; ordinary rescan/add does neither.
c=create(); c.status.catalog.sources=[{directory:'/synthetic/first'},{directory:'/synthetic/second'}];
c.reindexAll(true); current=c.commands.shift();
assert.equal(current.command.directory,'/synthetic/first'); assert.equal(current.command.reindexText,true); assert.equal(current.command.verifyContent,true);
c.receive(JSON.stringify({id:current.id,ok:true,result:{failed:0,failures:[],cancelled:false,indexed:1,unchanged:0,removed:0}}));
current=c.commands.shift(); assert.equal(current.command.directory,'/synthetic/second'); assert.equal(current.command.reindexText,true);
c.cancelIndex(); current=c.commands.shift(); assert.equal(current.command.type,'cancel'); assert.equal(c.reindexQueue.length,0);
c=create(); c.status.catalog.sources=[{directory:'/synthetic/ordinary'}]; c.reindexAll(); current=c.commands.shift();
assert.equal(current.command.type,'index'); assert.equal(current.command.reindexText,undefined); assert.equal(current.command.verifyContent,undefined);
c=create(); c.indexFolder('/synthetic/new'); current=c.commands.shift(); assert.equal(current.command.reindexText,undefined);
console.log('PASS: actual Controller JS media/query/folder scope, bounded coalescing, model-free browsing, hybrid order/labels, debounce, warmup/reconnect and stale previews');
console.log('PASS: grouped passage bounds/order/location/actions, segment-mode preservation, grouping refresh, live idle settings and explicit versus ordinary indexing');
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
