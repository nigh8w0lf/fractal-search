#!/usr/bin/env python3
"""Real startup model preparation, detach/reconnect and shutdown; temporary catalogs only."""
import argparse
import json
from pathlib import Path
import tempfile
import time
from smoke_plugin import Client, launch, stop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', type=Path)
    parser.add_argument('models', type=Path)
    args = parser.parse_args()
    binary, models = args.binary.resolve(), args.models.resolve()
    with tempfile.TemporaryDirectory(prefix='fractal-search-startup-', dir='/tmp') as temporary:
        root = Path(temporary)
        for name, cache in [('real', models), ('missing', root/'missing-models'), ('corrupt', root/'corrupt-models')]:
            case = root/name; case.mkdir()
            if name == 'corrupt':
                cache.mkdir()
                manifest = json.loads((Path(__file__).resolve().parents[2]/'crates/fractal-search-core/models/embeddinggemma2.json').read_text())
                for asset in manifest['files']:
                    (cache/asset['path']).write_bytes(b'corrupt')
            process, client, path, log = launch(binary, cache, case, warm_search=True)
            try:
                client.send(1, {'type':'status'})
                result = client.response(1)['result']
                if name == 'real':
                    assert result['textModelLoaded'] and not result['textModelWarming'], result
                    children = (Path('/proc')/str(process.pid)/'task'/str(process.pid)/'children').read_text().split()
                    for pid in [str(process.pid), *children]:
                        memory = [line for line in (Path('/proc')/pid/'status').read_text().splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))]
                        print('Text warmup resident memory', pid, '; '.join(memory), flush=True)
                    client.close(); time.sleep(.1)
                    client = Client(path)
                    ready = client.until(lambda m:m.get('event')=='ready')['payload']
                    assert ready['textModelLoaded'] and not ready['textModelWarming'], ready
                    client.send(2, {'type':'search','query':'a red bicycle'})
                    events = []
                    result = client.until(lambda m: events.append(m) is None and m.get('id')==2 and 'ok' in m)
                    assert result['ok'] and result['result']['total']==0, result
                    assert not any(m.get('event')=='searchProgress' and m['payload']['stage']=='starting' for m in events), events
                    assert 'Loaded onnx/vision_encoder_fp16.onnx' not in (case/'daemon.log').read_text()
                elif name == 'missing':
                    assert not result['modelsPresent'] and not result['textModelWarming'] and not result['textModelLoaded'], result
                    assert not cache.exists(), 'Startup created/downloaded models'
                else:
                    assert not result['textModelLoaded'] and not result['textModelWarming'], result
                    client.close(); time.sleep(.1)
                    client = Client(path)
                    ready = client.until(lambda m:m.get('event')=='ready')['payload']
                    assert 'Warm up text search' in ready['warmupError'], ready
                    assert 'warmup failed' in (case/'daemon.log').read_text()
                client.send(3, {'type':'shutdown'}); client.response(3)
                assert process.wait(timeout=30)==0
            finally:
                client.close(); stop(process); log.close()
        # Stop requested while verification/loading is active must still drain.
        case=root/'stop-during-warmup'; case.mkdir()
        process, client, path, log=launch(binary, models, case, warm_search=True)
        try:
            client.send(1, {'type':'shutdown'}); client.response(1)
            assert process.wait(timeout=30)==0
        finally:
            client.close(); stop(process); log.close()
    print('Startup: real text-only warmup, warm first search, reconnect, visible corruption, missing-model setup and graceful stop passed')


if __name__ == '__main__':
    main()
