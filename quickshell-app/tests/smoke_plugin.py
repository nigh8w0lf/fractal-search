#!/usr/bin/env python3
"""Plugin transport tests: only temporary catalogs, sockets and synthetic images."""
import argparse
import json
import os
from pathlib import Path
import queue
import socket
import stat
import subprocess
import tempfile
import threading
import time
from smoke_backend import png


class Client:
    def __init__(self, path):
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.connect(str(path))
        self.messages = queue.Queue()
        self.file = self.socket.makefile('r')
        def read():
            try:
                for line in self.file:
                    self.messages.put(json.loads(line))
            except Exception as error:
                self.messages.put(error)
            finally:
                self.messages.put(EOFError('Socket closed'))
        threading.Thread(target=read, daemon=True).start()

    def send(self, request_id, command):
        self.socket.sendall((json.dumps({'id': request_id, 'command': command}) + '\n').encode())

    def until(self, predicate, timeout=90):
        deadline = time.monotonic() + timeout
        while True:
            message = self.messages.get(timeout=max(.01, deadline - time.monotonic()))
            if isinstance(message, Exception):
                raise message
            if predicate(message):
                return message

    def response(self, request_id, success=True):
        result = self.until(lambda m: m.get('id') == request_id and 'ok' in m)
        assert result['ok'] is success, result
        return result

    def close(self):
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        self.file.close()


def launch(binary, models, root, env=None, warm_search=False):
    path = root / 'run/backend.sock'
    log = (root / 'daemon.log').open('a')
    command = [str(binary), 'daemon', '--socket', str(path), '--data-dir', str(root / 'data'), '--model-dir', str(models)]
    if warm_search:
        command.append('--warm-search')
    process = subprocess.Popen(command, stderr=log, stdout=subprocess.DEVNULL, env=env)
    deadline = time.monotonic() + 10
    while not path.exists():
        assert process.poll() is None, (root / 'daemon.log').read_text()
        assert time.monotonic() < deadline
        time.sleep(.05)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    client = Client(path)
    ready = client.until(lambda m: m.get('event') == 'ready')['payload']
    assert ready['protocolVersion'] == 1
    return process, client, path, log


def stop(process):
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', type=Path)
    parser.add_argument('models', type=Path)
    args = parser.parse_args()
    binary, models = args.binary.resolve(), args.models.resolve()
    with tempfile.TemporaryDirectory(prefix='fractal-search-plugin-', dir='/tmp') as temp:
        root = Path(temp)
        process, client, path, log = launch(binary, models, root)
        try:
            # Duplicate clients get an explicit error and do not kill the service.
            duplicate = Client(path)
            assert 'already' in duplicate.until(lambda m: m.get('event') == 'fatal')['payload']['error']
            duplicate.close()
            client.send(1, {'type': 'installModels'})
            assert client.response(1)['result']['modelsPresent']
            print('Verified existing model cache without redownload', flush=True)
            images = root / 'photos'; images.mkdir()
            for i in range(5):
                png(images / f'{i}.png', color=(i * 30, 80, 180))
            (images / 'bad.jpg').write_bytes(b'broken jpeg')
            client.send(2, {'type': 'index', 'directory': str(images)})
            client.until(lambda m: m.get('event') == 'indexProgress')
            client.close()
            time.sleep(.1)
            client = Client(path)
            snapshot = client.until(lambda m: m.get('event') == 'ready')['payload']
            assert snapshot['activeJob']['kind'] == 'index', snapshot
            # Reusing the old client's ID must not consume its index result.
            client.send(2, {'type': 'status'})
            finished = client.until(lambda m: m.get('event') == 'indexFinished')['payload']
            assert finished['indexed'] == 5 and finished['failed'] == 1, finished
            assert client.response(2)['result']['catalog']['imageCount'] == 5
            print('UI disconnect/reconnect preserved indexing, IDs and skipped-file report', flush=True)
            client.close(); time.sleep(.1)
            client = Client(path)
            snapshot = client.until(lambda m: m.get('event') == 'ready')['payload']
            assert snapshot['lastIndexReport']['failed'] == 1
            client.send(3, {'type': 'search', 'query': 'blue image', 'limit': 5, 'maxResults': 5, 'offset': 0, 'sourceId': None})
            result = client.response(3)['result']
            assert result['total'] == 5
            client.send(4, {'type': 'preview', 'assetId': result['items'][0]['assetId']})
            assert client.response(4)['result']['url'].startswith('file:')
            client.send(5, {'type': 'status', 'unknown': 1})
            assert client.until(lambda m: m.get('id') is None and m.get('ok') is False)['error']
            client.send(6, {'type': 'shutdown'})
            client.response(6)
            assert client.until(lambda m: m.get('event') == 'stopped')['payload']['clean']
            assert process.wait(timeout=15) == 0
            client.close()
            print('Real ONNX search/preview, strict requests and graceful stop passed', flush=True)
            process, client, path, log = launch(binary, models, root)
            for i in range(5, 10):
                png(images / f'{i}.png', color=(80, i * 10, 40))
            client.send(1, {'type': 'index', 'directory': str(images)})
            client.until(lambda m: m.get('event') == 'indexProgress' and m['payload']['stage'] != 'scanning')
            process.terminate()
            assert process.wait(timeout=90) == 0
            client.close()
            process, client, path, log = launch(binary, models, root)
            client.send(1, {'type': 'status'})
            assert client.response(1)['result']['catalog']['imageCount'] >= 5
            client.send(2, {'type': 'shutdown'}); client.response(2)
            assert process.wait(timeout=15) == 0
            client.close()
            print('SIGTERM drains active image and catalog reopens safely', flush=True)
        finally:
            stop(process); log.close()
        # Missing models: exercise download progress and cancellation offline.
        empty = root / 'empty-models'; empty.mkdir()
        fakebin = root / 'fakebin'; fakebin.mkdir()
        curl = fakebin / 'curl'
        curl.write_text('#!/usr/bin/python3\nimport sys,time\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index("--output")+1]).write_bytes(b"partial")\ntime.sleep(120)\n')
        curl.chmod(0o755)
        cancel = root / 'cancel'; cancel.mkdir()
        environment = dict(os.environ, PATH=str(fakebin) + ':' + os.environ['PATH'])
        process, client, path, log = launch(binary, empty, cancel, environment)
        try:
            client.send(1, {'type': 'installModels'})
            progress = client.until(lambda m: m.get('event') == 'modelProgress' and m['payload']['stage'] == 'downloading')
            assert progress['payload']['total'] > 1_000_000_000
            client.send(2, {'type': 'shutdown'})
            assert 'stopped' in client.response(1, False)['error']
            client.response(2)
            assert process.wait(timeout=15) == 0
            client.close()
            assert not (empty / 'visual.onnx').exists()
            print('Offline model-download progress/cancel leaves no unverified asset', flush=True)
        finally:
            stop(process); log.close()
    print('Plugin smoke checks passed')


if __name__ == '__main__':
    main()
