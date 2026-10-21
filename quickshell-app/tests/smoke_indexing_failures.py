#!/usr/bin/env python3
"""Opt-in decoder error classification checks using synthetic media and a temporary catalog."""

import argparse
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import tempfile
import wave

from smoke_backend import Service, png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('models', type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    models = args.models.resolve(strict=True)
    probe_executable = shutil.which('ffprobe')
    assert probe_executable, 'This check requires ffprobe'
    ffprobe = Path(probe_executable).resolve(strict=True)
    decoder_executable = shutil.which('ffmpeg')
    assert decoder_executable, 'This check requires ffmpeg'
    ffmpeg = Path(decoder_executable).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='fractal-failure-smoke-', dir='/tmp') as tmp:
        root = Path(tmp)
        source = root / 'source'
        source.mkdir()
        png(source / 'healthy.png')
        audio = source / 'tone.wav'
        with wave.open(str(audio), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b''.join(struct.pack('<h', int(8000 * math.sin(2 * math.pi * 440 * i / 16000))) for i in range(16000)))
        data = root / 'data'
        service = Service(binary, models, data, root / 'seed.log')
        try:
            service.until(lambda item: item.get('event') == 'ready')
            request = service.send({'type': 'index', 'directory': str(source)})
            service.until(lambda item: item.get('id') == request and 'ok' in item, timeout=300)
            result = service.response(request)
            assert (result['indexed'], result['failed']) == (2, 0), result
            baseline = service.call({'type': 'browse', 'limit': 100})
        finally:
            service.close()
        # Force a changed source while keeping a valid WAV. Every failed rescan
        # must retain the previously indexed rows rather than deleting them.
        with audio.open('ab') as output:
            output.write(b'\0\0')
        cases = [
            ('missing-probe', None, None, 'install ffmpeg and ffprobe', True),
            ('missing-decoder', 'real', None, 'install ffmpeg and ffprobe', True),
            ('decoder-crash', 'real', 'kill -KILL $$', 'terminated abnormally', True),
            ('bad-protocol', "printf 'not JSON'", None, 'malformed metadata JSON', True),
            ('bad-input', 'real', "printf 'damaged fixture input' >&2; exit 1", 'damaged fixture input', False),
        ]
        original_path = os.environ['PATH']
        for name, probe, decoder, diagnostic, fatal in cases:
            if not fatal:
                png(source / 'z-after-failure.png', color=(10, 60, 200))
            commands = root / name
            commands.mkdir()
            for tool, script in [('ffprobe', probe), ('ffmpeg', decoder)]:
                if script == 'real':
                    (commands / tool).symlink_to(ffprobe)
                elif script is not None:
                    executable = commands / tool
                    executable.write_text('#!/bin/sh\n' + script + '\n')
                    executable.chmod(0o700)
            # Only this child receives the fixture PATH; the installed service
            # and system binaries are never modified.
            try:
                os.environ['PATH'] = str(commands)
                service = Service(binary, models, data, root / (name + '.log'))
            finally:
                os.environ['PATH'] = original_path
            try:
                service.until(lambda item: item.get('event') == 'ready')
                request = service.send({'type': 'index', 'directory': str(source)})
                service.until(lambda item: item.get('id') == request and 'ok' in item, timeout=300)
                result = service.response(request, success=not fatal)
                if fatal:
                    assert 'Media pipeline failed' in result, result
                    assert diagnostic in result, result
                else:
                    assert (result['failed'], result['unchanged'], result['indexed']) == (1, 1, 1), result
                    assert diagnostic in result['failures'][0]['error'], result
                current = service.call({'type': 'browse', 'limit': 100})
                if fatal:
                    assert current == baseline, name
                else:
                    assert len(current['items']) == 3, current
                    retained = {item['assetId']: item for item in current['items']}
                    assert all(retained[item['assetId']] == item for item in baseline['items']), current
                print(name + ': passed; prior rows preserved', flush=True)
            finally:
                service.close()
        # Exercise the segment-level catch after visuals have succeeded. A crash
        # in the audio subprocess must abort instead of publishing partial rows.
        video_source = root / 'video-source'
        video_source.mkdir()
        video = video_source / 'movie.mkv'
        subprocess.run([str(ffmpeg), '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                        'color=c=red:s=32x24:r=1:d=2', '-f', 'lavfi', '-i',
                        'sine=frequency=440:sample_rate=16000:duration=2',
                        '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(video)], check=True)
        service = Service(binary, models, data, root / 'video-seed.log')
        try:
            service.until(lambda item: item.get('event') == 'ready')
            request = service.send({'type': 'index', 'directory': str(video_source)})
            service.until(lambda item: item.get('id') == request and 'ok' in item, timeout=300)
            result = service.response(request)
            assert (result['indexed'], result['failed']) == (1, 0), result
            baseline = service.call({'type': 'browse', 'limit': 100})
        finally:
            service.close()
        content = video.read_bytes()
        old_duration = b'00:00:02.000000000'
        offset = content.rindex(old_duration)
        video.write_bytes(content[:offset] + b'00:00:03.000000000' + content[offset + len(old_duration):])
        commands = root / 'soundtrack-crash'
        commands.mkdir()
        (commands / 'ffprobe').symlink_to(ffprobe)
        executable = commands / 'ffmpeg'
        executable.write_text('#!/bin/sh\nfor arg do\n'
                              '  if [ "$arg" = pcm_f32le ]; then kill -KILL $$; fi\ndone\n'
                              + 'exec ' + shlex.quote(str(ffmpeg)) + ' "$@"\n')
        executable.chmod(0o700)
        try:
            os.environ['PATH'] = str(commands)
            service = Service(binary, models, data, root / 'soundtrack-crash.log')
        finally:
            os.environ['PATH'] = original_path
        try:
            service.until(lambda item: item.get('event') == 'ready')
            request = service.send({'type': 'index', 'directory': str(video_source)})
            service.until(lambda item: item.get('id') == request and 'ok' in item, timeout=300)
            error = service.response(request, success=False)
            assert 'terminated abnormally' in error and 'Decode audio segment' in error, error
            assert service.call({'type': 'browse', 'limit': 100}) == baseline
            print('soundtrack crash: fatal; previous visual/audio rows preserved', flush=True)
        finally:
            service.close()
        print('Decoder classification and failed-replacement preservation passed', flush=True)


if __name__ == '__main__':
    main()
