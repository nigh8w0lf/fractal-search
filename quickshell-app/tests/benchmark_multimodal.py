#!/usr/bin/env python3
"""Compare warmed mixed-media indexing using temporary synthetic files only."""
import argparse,json,time,os,tempfile
from pathlib import Path
from smoke_multimodal import fixtures
from smoke_backend import Service
from benchmark_indexing import cpu_seconds,index
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('binary', type=Path)
parser.add_argument('models', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--compare', type=Path)
args=parser.parse_args()
binary,models,output=args.binary.resolve(),args.models.resolve(),args.output.resolve()
output.mkdir()
with tempfile.TemporaryDirectory(prefix='fractal-pool-mixed-') as tmp:
    root=Path(tmp)
    for name in ['warmup','media','data']:(root/name).mkdir()
    fixtures(root/'warmup');fixtures(root/'media')
    settings={'modelDirectory':str(models),'ui':{'topMatches':6,'resultLayout':'listPreview','watchChanges':False,'closeAfterOpening':True,'imageBatchSize':2}}
    (root/'data/settings.json').write_text(json.dumps(settings))
    service=Service(binary,models,root/'data',output/'backend.log')
    try:
        ready=service.until(lambda e:e.get('event')=='ready')['payload']
        assert index(service,root/'warmup')['indexed']==5
        begin=time.monotonic();cpu=cpu_seconds(service.process.pid)
        result=index(service,root/'media');elapsed=time.monotonic()-begin;cpu=cpu_seconds(service.process.pid)-cpu
        assert result['indexed']==5 and result['failed']==0 and not result['cancelled'],result
        source=next(s['id'] for s in service.call({'type':'status'})['catalog']['sources'] if s['directory']==str(root/'media'))
        hits=service.call({'type':'search','query':'a red bicycle near a cafe','sourceId':source,'maxResults':100,'limit':100})['items']
        scores={f"{x['filename']}:{x['mediaKind']}:{x['segmentIndex']}":x['similarity'] for x in hits}
        report={'binary':str(binary),'modelProfile':ready['catalog']['modelProfile'],'elapsedSeconds':elapsed,'cpuSeconds':cpu,'scores':scores,'provider':ready['provider'],'deviceId':ready['deviceId'],'threads':ready['threads'],'affinity':sorted(os.sched_getaffinity(0)),'note':'One warmed scan of five synthetic mixed files; same text passages reuse cached vectors after warmup. Not a text throughput or large-library benchmark.'}
        if args.compare:
            old=json.loads(args.compare.read_text())
            for key in ['provider','deviceId','threads','affinity','modelProfile']:assert old[key]==report[key],key
            assert old['scores'].keys()==scores.keys()
            error=max(abs(old['scores'][k]-v) for k,v in scores.items());assert error<=1e-4,error
            report['comparison']={'maximumScoreError':error,'throughputRatio':old['elapsedSeconds']/elapsed,'scoresCompared':len(scores)}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='scores'},indent=2),flush=True)
    finally:service.close()
