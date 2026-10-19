#!/usr/bin/env python3
"""Opt-in real multimodal inference and read-only MCP checks; synthetic temporary data only."""
import argparse
import base64
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import wave

from smoke_backend import Service, local_url, png
from smoke_mcp import Mcp, catalog_hashes, data


def fixtures(directory):
    png(directory / 'red.png', 96, 64, (220, 20, 10))
    (directory / 'notes.md').write_text(('A red bicycle is parked by the café. Unicode passage.\n' * 150), encoding='utf-8')
    # Cross the 30-second boundary without large source fixtures or downloads.
    with wave.open(str(directory / 'tone.wav'), 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        for second in range(31):
            audio.writeframes(b''.join(struct.pack('<h', int(8000 * math.sin(2 * math.pi * 440 * (second * 16000 + i) / 16000))) for i in range(16000)))
    for soundtrack, name in [(True, 'clip.mp4'), (False, 'silent.mp4')]:
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=96x64:r=2:d=2']
        if soundtrack:
            command += ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000:duration=2']
        command += ['-t', '2', '-c:v', 'mpeg4', '-q:v', '5']
        command += ['-c:a', 'aac', '-shortest'] if soundtrack else ['-an']
        subprocess.run([*command, str(directory / name)], check=True, capture_output=True, timeout=30)


def checked_segments(items, desktop):
    assert {item['mediaKind'] for item in items} == {'image', 'text', 'audio', 'video'}
    for item in items:
        assert item['assetId'].startswith('asset_') and item['segmentIndex'] >= 0
        assert not {'indexedText', 'indexed_text'} & item.keys()
        assert item['textExcerpt'] is None or len(item['textExcerpt'].encode('utf-8')) <= 2048
        if item['mediaKind'] in ('image', 'video'):
            assert local_url(item['thumbnailUrl']).is_file()
            assert item['width'] > 0 and item['height'] > 0
        else:
            assert item['thumbnailUrl'] == '' and item['width'] == item['height'] == 0
            assert desktop.call({'type': 'preview', 'assetId': item['assetId']}, success=False)
        if item['mediaKind'] in ('audio', 'video'):
            assert 0 <= item['startMs'] < item['endMs']
            assert item['endMs'] - item['startMs'] <= 30000
        else:
            assert item['startMs'] is None and item['endMs'] is None
        assert local_url(desktop.call({'type': 'originalUrl', 'assetId': item['assetId']})['url']).is_file()
    document = [item for item in items if item['filename'] == 'notes.md']
    assert len(document) >= 2 and all(item['mediaKind'] == 'text' and item['textExcerpt'] for item in document)
    tone = sorted((item for item in items if item['filename'] == 'tone.wav'), key=lambda item: item['segmentIndex'])
    assert len(tone) == 2 and [item['startMs'] for item in tone] == [0, 30000]
    assert {item['mediaKind'] for item in items if item['filename'] == 'clip.mp4'} == {'video', 'audio'}
    assert {item['mediaKind'] for item in items if item['filename'] == 'silent.mp4'} == {'video'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', type=Path)
    parser.add_argument('mcp', type=Path)
    parser.add_argument('models', type=Path)
    args = parser.parse_args()
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            parser.error(f'{tool} is required; install FFmpeg explicitly before running this test')
    with tempfile.TemporaryDirectory(prefix='fractal-multimodal-') as temporary:
        root = Path(temporary)
        originals = root / 'originals'; originals.mkdir()
        fixtures(originals)
        catalog = root / 'data'
        desktop = Service(args.backend.resolve(), args.models.resolve(), catalog, root / 'desktop.log')
        mcp = None
        try:
            ready = desktop.until(lambda event: event.get('event') == 'ready')['payload']
            assert ready['catalog']['fileCount'] == ready['catalog']['assetCount'] == 0
            assert ready['catalog']['modelProfile'] == 'embeddinggemma2-daa72c51-fp16-multimodal-mrl512-v3'
            request = desktop.send({'type': 'index', 'directory': str(originals)})
            # Multimodal inference can take minutes on CPU.
            desktop.until(lambda event: event.get('id') == request and 'ok' in event, timeout=1800)
            indexed = desktop.response(request)
            assert indexed['indexed'] == 5 and indexed['failed'] == 0, indexed
            items = desktop.call({'type': 'browse', 'limit': 100})['items']
            checked_segments(items, desktop)
            count = len(items)
            status = desktop.call({'type': 'status'})['catalog']
            assert status['fileCount'] == 5 and status['assetCount'] == count > 5, status
            assert status['sources'][0]['fileCount'] == 5 and status['sources'][0]['assetCount'] == count
            # Explicit filters preserve ordinary browse ordering and source scope.
            source_id = status['sources'][0]['id']
            for kind in ('image', 'text', 'audio', 'video'):
                expected = [item for item in items if item['mediaKind'] == kind]
                filtered = desktop.call({'type': 'browse', 'mediaKind': kind, 'sourceId': source_id, 'limit': 100})
                assert filtered['total'] == len(expected) and filtered['items'] == expected
            for kind in ('all', 'Text', 'document'):
                for command in ({'type': 'browse', 'mediaKind': kind}, {'type': 'search', 'query': 'bicycle', 'mediaKind': kind}):
                    desktop.send(command)
                    rejected = desktop.until(lambda event: event.get('id') is None and event.get('ok') is False)
                    assert 'unknown variant' in rejected['error'], rejected
            query = 'a red bicycle near a café'
            all_hits = desktop.call({'type': 'search', 'query': query, 'maxResults': 100, 'limit': 100})
            assert all_hits['searchMode'] == 'semantic'
            assert all(item.get('hybridScore') is None for item in all_hits['items'])
            assert [item['similarity'] for item in all_hits['items']] == sorted((item['similarity'] for item in all_hits['items']), reverse=True)
            for kind in ('text', 'image', 'audio', 'video'):
                filtered = desktop.call({'type': 'search', 'query': query, 'mediaKind': kind, 'sourceId': source_id, 'maxResults': 100, 'limit': 100})
                assert filtered['items'] and all(item['mediaKind'] == kind for item in filtered['items'])
                assert filtered['searchMode'] == ('hybrid' if kind == 'text' else 'semantic')
                assert all(math.isfinite(item['similarity']) for item in filtered['items'])
                if kind == 'text':
                    assert all(math.isfinite(item['hybridScore']) and 0 < item['hybridScore'] <= 2 / 61 for item in filtered['items'])
                    assert [item['hybridScore'] for item in filtered['items']] == sorted((item['hybridScore'] for item in filtered['items']), reverse=True)
                else:
                    assert [item['assetId'] for item in filtered['items']] == [item['assetId'] for item in all_hits['items'] if item['mediaKind'] == kind]
            # Identical query/limit cannot reuse a text-filtered search cache for All media.
            repeated = desktop.call({'type': 'search', 'query': query, 'maxResults': 100, 'limit': 100})
            assert repeated == all_hits
            old_ids = {item['assetId'] for item in items}
            assert desktop.call({'type': 'index', 'directory': str(originals)})['unchanged'] == 5
            assert {item['assetId'] for item in desktop.call({'type': 'browse', 'limit': 100})['items']} == old_ids
            # A failed replacement must preserve the previous complete document.
            (originals / 'notes.md').write_bytes(b'\xff invalid UTF-8')
            failed = desktop.call({'type': 'index', 'directory': str(originals)})
            assert failed['failed'] == 1 and failed['indexed'] == 0, failed
            assert {item['assetId'] for item in desktop.call({'type': 'browse', 'limit': 100})['items']} == old_ids
            shutil.rmtree(originals)
            before = catalog_hashes(catalog)
            mcp = Mcp(args.mcp.resolve(), catalog, args.models.resolve(), root / 'mcp.log')
            status = data(mcp.tool('status'))
            assert status['assetCount'] == count and status['fileCount'] == 5 and not status['textModelLoaded']
            assert status['modelProfile'] == 'embeddinggemma2-daa72c51-fp16-multimodal-mrl512-v3'
            page = data(mcp.tool('browse_assets', {'limit': 100}))
            assert page['total'] == count
            for item in page['items']:
                assert data(mcp.tool('get_asset_metadata', {'assetId': item['assetId']})) == item
                assert not {'path', 'directory', 'thumbnailUrl', 'originalUrl', 'indexedText', 'indexed_text'} & item.keys()
                if item['mediaKind'] in ('text', 'audio'):
                    assert item['thumbnailHandle'] is None
                    missing = mcp.tool('get_thumbnail', {'thumbnailHandle': 'thumbnail_' + item['assetId']}, success=False)
                    assert missing['result']['isError'], missing
                else:
                    block = mcp.tool('get_thumbnail', {'thumbnailHandle': item['thumbnailHandle']})['result']['content'][0]
                    assert block['mimeType'] == 'image/webp'
                    assert len(base64.b64decode(block['data'], validate=True)) <= 1024 * 1024
            for kind in ('image', 'text', 'audio', 'video'):
                filtered = data(mcp.tool('browse_assets', {'mediaKind': kind, 'limit': 100}))
                expected = [item for item in page['items'] if item['mediaKind'] == kind]
                assert filtered['total'] == len(expected) and filtered['items'] == expected
            assert not data(mcp.tool('status'))['textModelLoaded'], 'Filtered browsing must remain model-free'
            for kind in ('all', 'Text', 'document'):
                assert 'error' in mcp.tool('browse_assets', {'mediaKind': kind}, success=False)
                assert 'error' in mcp.tool('search_assets', {'query': 'bicycle', 'mediaKind': kind}, success=False)
            hits = data(mcp.tool('search_assets', {'query': 'a red bicycle near a café', 'limit': 100}))['items']
            assert len(hits) == count
            assert all(math.isfinite(item['similarity']) for item in hits)
            assert {item['mediaKind'] for item in hits} == {'image', 'text', 'audio', 'video'}
            assert all(item.get('hybridScore') is None for item in hits)
            assert [item['similarity'] for item in hits] == sorted((item['similarity'] for item in hits), reverse=True)
            hybrid = data(mcp.tool('search_assets', {'query': query, 'mediaKind': 'text', 'limit': 100}))
            assert hybrid['searchMode'] == 'hybrid' and hybrid['items']
            assert all(item['mediaKind'] == 'text' and math.isfinite(item['similarity']) for item in hybrid['items'])
            assert all(math.isfinite(item['hybridScore']) and 0 < item['hybridScore'] <= 2 / 61 for item in hybrid['items'])
            assert [item['hybridScore'] for item in hybrid['items']] == sorted((item['hybridScore'] for item in hybrid['items']), reverse=True)
            assert not any('indexedText' in item or 'indexed_text' in item for item in hybrid['items'])
            semantic_again = data(mcp.tool('search_assets', {'query': query, 'limit': 100}))
            assert semantic_again['searchMode'] == 'semantic' and semantic_again['items'] == hits

            assert not any(str(root) in json.dumps(message) or 'file://' in json.dumps(message) for message in mcp.history)
            mcp.stop(); mcp = None
            assert catalog_hashes(catalog) == before
            assert 'Loaded onnx/vision_encoder_fp16.onnx' not in (root / 'mcp.log').read_text()
            assert 'Loaded onnx/audio_encoder_fp16.onnx' not in (root / 'mcp.log').read_text()
            desktop.close()
        finally:
            if mcp is not None:
                mcp.process.terminate(); mcp.process.wait(timeout=30); mcp.log.close()
            if desktop.process.poll() is None:
                desktop.process.terminate(); desktop.process.wait(timeout=120)
            desktop.log.close()
    print('PASS: image/text/audio/video segments, video soundtrack, bounds, stable rescans, atomic failed replacement, all-media cosine, text-only hybrid, explicit scope and cached read-only MCP with originals removed')


if __name__ == '__main__':
    main()
