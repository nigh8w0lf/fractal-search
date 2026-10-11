#!/usr/bin/env python3
"""Measure a warmed synthetic image scan without touching the user's catalog."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import tempfile
import time
import zlib

from smoke_backend import Service, png


def patterned_png(path, index):
    width, height = 1024, 1536
    scanlines = bytearray()
    for y in range(height):
        scanlines.append(0)
        scanlines.extend(bytes((x * 7 + y * 3 + index * 23) % 256 for x in range(width * 3)))

    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))

    path.write_bytes(b'\x89PNG\r\n\x1a\n'
                     + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))


def cpu_seconds(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')


def index(service, folder):
    request = service.send({'type': 'index', 'directory': str(folder)})
    service.until(lambda event: event.get('id') == request and 'ok' in event, timeout=300)
    return service.response(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('models', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--compare', type=Path, help='Previous report for score parity and timing comparison')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='fractal-index-benchmark-', dir='/tmp') as temporary:
        root = Path(temporary)
        warmup, images, data = (root / name for name in ['warmup', 'images', 'data'])
        for folder in [warmup, images, data]:
            folder.mkdir()
        for i in range(2):
            png(warmup / f'{i}.png')
        for i in range(8):
            patterned_png(images / f'{i}.png', i)
        preferences = {'topMatches': 6, 'resultLayout': 'listPreview', 'watchChanges': False,
                       'closeAfterOpening': True, 'imageBatchSize': 2}
        (data / 'settings.json').write_text(json.dumps({'modelDirectory': str(args.models.resolve()), 'ui': preferences}))
        log = root / 'backend.log'
        service = Service(args.binary.resolve(), args.models.resolve(), data, log)
        try:
            ready = service.until(lambda event: event.get('event') == 'ready')['payload']
            assert ready['activeImageBatchSize'] == 2
            assert index(service, warmup)['indexed'] == 2
            print('Warmup complete; measuring eight synthetic 1024x1536 images at batch2', flush=True)
            log_offset = log.stat().st_size
            initial_cpu = cpu_seconds(service.process.pid)
            began = time.monotonic()
            result = index(service, images)
            elapsed = time.monotonic() - began
            cpu = cpu_seconds(service.process.pid) - initial_cpu
            assert result['indexed'] == 8 and result['failed'] == 0 and not result['cancelled'], result
            measured_log = log.read_bytes()[log_offset:].decode('utf-8')
            stages = {}
            for field in ['decode_ms', 'prepare_and_load_ms', 'inference_ms', 'thumbnail_and_store_ms']:
                values = [float(value) for value in re.findall(r'\b' + field + r'=([0-9.]+)', measured_log)]
                assert values, 'Missing timing field: ' + field
                expected = 4 if field in ['prepare_and_load_ms', 'inference_ms'] else 8
                assert len(values) == expected, 'Unexpected timed image/batch count: ' + field
                stages[field] = {'count': len(values), 'sumMs': sum(values), 'medianMs': statistics.median(values)}
            sources = service.call({'type': 'status'})['catalog']['sources']
            source = next(item['id'] for item in sources if item['directory'] == str(images))
            queries = ['a colorful portrait', 'green stripes', 'blue landscape']
            scores = {}
            for query in queries:
                hits = service.call({'type': 'search', 'query': query, 'sourceId': source, 'maxResults': 8, 'limit': 8})
                assert len(hits['items']) == 8 and hits['searchMode'] == 'semantic'
                scores[query] = {item['filename']: item['similarity'] for item in hits['items']}
                assert all(math.isfinite(value) for value in scores[query].values())
            report = {'binary': str(args.binary.resolve()), 'provider': ready['provider'], 'deviceId': ready['deviceId'],
                      'threads': ready['threads'],
                      'imageBatchSize': 2, 'images': 8, 'dimensions': [1024, 1536], 'elapsedSeconds': elapsed,
                      'cpuSeconds': cpu, 'cpuPercentOneCore': 100 * cpu / elapsed, 'imagesPerSecond': 8 / elapsed,
                      'cpuAffinity': sorted(os.sched_getaffinity(0)), 'nice': os.nice(0), 'stages': stages, 'scores': scores,
                      'timingNote': 'One warmed synthetic scan; identical image fixture and batch size. CPU affinity/nice are recorded; no concurrent compilation. Different binaries may also include compiler/build differences.'}
            if args.compare:
                previous = json.loads(args.compare.read_text())
                for setting in ['provider', 'deviceId', 'threads', 'imageBatchSize', 'images', 'dimensions', 'cpuAffinity', 'nice']:
                    assert previous[setting] == report[setting], 'Benchmark setting changed: ' + setting
                errors = [abs(scores[query][name] - previous['scores'][query][name]) for query in queries for name in scores[query]]
                report['comparison'] = {'maximumScoreError': max(errors), 'scoresCompared': len(errors),
                                        'throughputRatio': previous['elapsedSeconds'] / elapsed,
                                        'cpuSecondsRatio': cpu / previous['cpuSeconds']}
                assert max(errors) <= 1e-4, report['comparison']
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({key: report[key] for key in ['elapsedSeconds', 'cpuSeconds', 'cpuPercentOneCore', 'imagesPerSecond']}, indent=2), flush=True)
            if args.compare:
                print(json.dumps(report['comparison'], indent=2), flush=True)
        finally:
            service.close()
            shutil.copyfile(log, args.output / 'backend.log')


if __name__ == '__main__':
    main()
