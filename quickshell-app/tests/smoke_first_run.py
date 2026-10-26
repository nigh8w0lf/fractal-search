#!/usr/bin/env python3
"""Bootstrap transactions using synthetic archives, Git clones and command fixtures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[1] / 'packaging/first-run.sh'

def main():
    with tempfile.TemporaryDirectory(prefix='fractal-first-run-') as directory:
        root = Path(directory)
        for case in ['success', 'cuda-path', 'cuda-invalid-path', 'local-success', 'local-wrong-upstream', 'local-changed-head', 'install-failure', 'checksum', 'dirty', 'wrong-origin', 'interrupted', 'launch', 'launch-failure']:
            work = root / case; work.mkdir()
            plugin = work / 'config/omarchy/plugins/fractal.search'; plugin.mkdir(parents=True)
            env = dict(os.environ, HOME=str(work), XDG_CONFIG_HOME=str(work/'config'), XDG_DATA_HOME=str(work/'data'), XDG_STATE_HOME=str(work/'state'), XDG_RUNTIME_DIR=str(work/'run'))
            subprocess.run(['git','init','-q',str(plugin)],check=True)
            (plugin/'manifest.json').write_text('{"id":"fractal.search"}')
            subprocess.run(['git','-C',str(plugin),'add','.'],check=True)
            subprocess.run(['git','-C',str(plugin),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','Fixture'],check=True)
            subprocess.run(['git','-C',str(plugin),'remote','add','origin','https://github.com/nigh8w0lf/fractal-search.git' if case!='wrong-origin' else 'https://example.invalid/wrong'],check=True)
            if case.startswith('local-'):
                origin = work / 'local-source'
                subprocess.run(['git', 'clone', '-q', str(plugin), str(origin)], check=True)
                subprocess.run(['git', '-C', str(origin), 'remote', 'set-url', 'origin',
                                'https://example.invalid/wrong' if case == 'local-wrong-upstream' else 'https://github.com/nigh8w0lf/fractal-search.git'], check=True)
                subprocess.run(['git', '-C', str(plugin), 'remote', 'set-url', 'origin', str(origin)], check=True)
                if case == 'local-changed-head':
                    subprocess.run(['git', '-C', str(origin), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '--allow-empty', '-qm', 'New head'], check=True)
            provider = 'cuda' if case.startswith('cuda-') else 'cpu'
            cuda_dir = work / 'private CUDA lib'
            cuda_dir.mkdir()
            bundle = work/f'fractal-search-0.1.1-linux-x86_64-{provider}'; bundle.mkdir()
            installer = bundle/'install.sh'
            installer.write_text('#!/bin/bash\nset -eu\n' + ('echo "Fixture missing dependency" >&2; exit 17\n' if case=='install-failure' else 'mkdir -p "$XDG_CONFIG_HOME/omarchy/plugins/fractal.search"\nprintf fixture > "$XDG_CONFIG_HOME/omarchy/plugins/fractal.search/.fractal-search-install"\n'))
            if case == 'cuda-path':
                installer.write_text(installer.read_text() + '[[ $1 == --provider && $2 == cuda && $3 == --cuda-lib-dir && $4 == "$HOME/private CUDA lib" ]]\n')
            installer.chmod(0o755)
            (bundle/'SHA256SUMS').write_text(hashlib.sha256(installer.read_bytes()).hexdigest()+'  install.sh\n')
            archive=work/'fixture.tar.gz'
            with tarfile.open(archive,'w:gz') as tar: tar.add(bundle,arcname=bundle.name)
            script=work/'first-run.sh'
            script.write_text(SOURCE.read_text().replace('41c228480393392f169d8ecddc6121ee103948c8fd33b47ae876d63409dc6de0',hashlib.sha256(archive.read_bytes()).hexdigest()).replace('a392124a0f09b8b005376479d4244ed4495044e69b77a8c32e4b5b1b1343da0c',hashlib.sha256(archive.read_bytes()).hexdigest() if case!='checksum' else '0'*64))
            commands=work/'commands';commands.mkdir()
            (commands/'curl').write_text('#!/bin/bash\nwhile (($#)); do if [[ $1 == --output ]]; then cp "$FIXTURE_ARCHIVE" "$2"; exit; fi; shift; done\nexit 1\n''')
            (commands/'systemctl').write_text('#!/bin/sh\necho inactive\n')
            (commands/'systemd-run').write_text('''#!/bin/bash\nprintf '%s\n' "$@" > "$FIXTURE_RUN_ARGS"\nif [[ $FIXTURE_CASE == launch-failure ]]; then exit 7; fi\nwhile (($#)); do if [[ $1 == -- ]]; then shift; "$@" >/dev/null 2>&1 & exit 0; fi; shift; done\nexit 1\n''')
            for file in commands.iterdir(): file.chmod(0o755)
            env.update(PATH=str(commands)+':'+env['PATH'],FIXTURE_ARCHIVE=str(archive),FIXTURE_CASE=case,FIXTURE_RUN_ARGS=str(work/'run-args'))
            if case=='dirty': (plugin/'untracked').write_text('keep me')
            if case=='interrupted':
                state=work/'state/fractal-search/setup';state.mkdir(parents=True)
                (state/'status.json').write_text('{"stage":"downloading","message":"old"}')
                os.utime(state/'status.json',(1,1))
                result=subprocess.run(['bash',str(script),'status'],env=env,capture_output=True,text=True)
                assert result.returncode==0 and json.loads(result.stdout)['stage']=='error',result
                print('PASS:',case)
                continue
            result=subprocess.run(['bash',str(script),'launch' if case.startswith('launch') else 'worker',provider,str(plugin),str(cuda_dir) if case == 'cuda-path' else ('/missing/cuda/path' if case == 'cuda-invalid-path' else '')],env=env,capture_output=True,text=True)
            if case=='launch':
                deadline=time.monotonic()+5
                while json.loads((work/'state/fractal-search/setup/status.json').read_text())['stage'] not in ['done','error']:
                    assert time.monotonic()<deadline
                    time.sleep(.05)
                args=(work/'run-args').read_text()
                assert '--unit=fractal-search-setup' in args and '--collect' in args
                assert str(work/'state/fractal-search/setup/worker.sh') in args
            state=json.loads((work/'state/fractal-search/setup/status.json').read_text())
            assert (result.returncode==0)==(case in ['success','cuda-path','local-success','launch']),(case,result,state)
            if case in ['success','cuda-path','local-success','launch']:
                assert state['stage']=='done'
                assert (plugin/'.fractal-search-install').is_file()
                assert list((work/'state/fractal-search/setup').glob('source-backup-*/.git'))
            else:
                assert state['stage']=='error' and (plugin/'.git').is_dir(),(case,state)
                assert not (plugin/'.fractal-search-install').exists()
            print('PASS:',case)
if __name__=='__main__': main()
