"""Maintainer-only mechanical refresh of the new native hook and source hashes.

Refuses an unrecognized prepared framework. Never used by a training launcher;
normal users prepare the already-pinned public patch with setup.py --framework.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
NATIVE = 'verl/trainer/ppo/rethink_external_router.py'


def sha(path, lf=False):
    value = path.read_bytes()
    return hashlib.sha256(value.replace(b'\r\n', b'\n') if lf else value).hexdigest()


def save(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))


def refresh():
    target = ROOT / '.deps/sdar'
    patch = ROOT / 'src/opsd/sdar.patch'
    upstream_path = ROOT / 'configs/upstream.json'
    lock_path = ROOT / 'configs/implementation_lock.json'
    upstream = json.loads(upstream_path.read_text())
    lock = json.loads(lock_path.read_text())
    receipt = json.loads((target / '.rethink_setup.json').read_text())
    if (sha(patch) != upstream['patch_sha256'] or
            any(receipt.get(k) != v for k, v in upstream.items()) or
            sha(target / 'verl/trainer/config/rethink_public.yaml') != receipt['config_sha256']):
        raise RuntimeError('Unrecognized prepared framework; no files overwritten')
    for name, wanted in lock['native_sha256_lf'].items():
        if sha(target / name, True) != wanted:
            raise RuntimeError('Prepared native source differs; no files overwritten')
    subprocess.run(['git', '-C', str(target), 'apply', '--reverse', '--check', str(patch)], check=True)
    text = patch.read_text(encoding='utf-8')
    marker = f'diff --git a/{NATIVE} b/{NATIVE}\n'
    if text.count(marker) != 1:
        raise RuntimeError('Native hook patch block is not unique')
    start = text.index(marker)
    end = text.find('diff --git ', start + len(marker))
    if end < 0:
        end = len(text)
    source = (ROOT / 'src/opsd/native_router.py').read_text(encoding='utf-8')
    lines = source.splitlines()
    block = marker + f'new file mode 100644\n--- /dev/null\n+++ b/{NATIVE}\n@@ -0,0 +1,{len(lines)} @@\n'
    block += ''.join('+' + line + '\n' for line in lines)
    patch.write_bytes((text[:start] + block + text[end:]).encode('utf-8'))
    (target / NATIVE).write_bytes(source.encode('utf-8'))
    shutil.copyfile(ROOT / 'configs/opsd.yaml', target / 'verl/trainer/config/rethink_public.yaml')
    upstream['patch_sha256'] = sha(patch)
    save(upstream_path, upstream)
    lock['source_sha256_lf']['src/opsd/native_router.py'] = ''
    for name in lock['source_sha256_lf']:
        lock['source_sha256_lf'][name] = sha(ROOT / name, True)
    for name in lock['native_sha256_lf']:
        lock['native_sha256_lf'][name] = sha(target / name, True)
    save(lock_path, lock)
    save(target / '.rethink_setup.json', {**upstream, 'config_sha256': sha(ROOT / 'configs/opsd.yaml')})
    subprocess.run(['git', '-C', str(target), 'apply', '--reverse', '--check', str(patch)], check=True)
    print('Native hook patch and implementation hashes refreshed; shared SDAR blocks unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', required=True)
    parser.parse_args()
    refresh()
