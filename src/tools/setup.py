"""Prepare pinned dependencies and bundled data; never start training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def git(root, *args):
    subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(root), *args], check=True)


def verify_framework(target, lock, patch, receipt):
    marker = target / '.rethink_setup.json'
    if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')) != receipt:
        raise RuntimeError('Dependency directory has a different setup; not overwritten')
    actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != lock['commit']:
        raise RuntimeError('SDAR HEAD differs from the pinned revision')
    git(target, 'apply', '--reverse', '--check', str(patch))
    if sha(target / 'verl/trainer/config/rethink_public.yaml') != receipt['config_sha256']:
        raise RuntimeError('Prepared SDAR configuration differs from the package')


def framework(local_source=None):
    lock = json.loads((ROOT / 'configs/upstream.json').read_text())
    patch = ROOT / 'src/opsd/sdar.patch'
    assert sha(patch) == lock['patch_sha256'], 'Patch differs from pinned package'
    target = ROOT / '.deps/sdar'
    receipt = dict(lock, config_sha256=sha(ROOT / 'configs/opsd.yaml'))
    if target.exists():
        verify_framework(target, lock, patch, receipt)
        print('Pinned SDAR dependency already prepared.')
        return
    target.parent.mkdir(exist_ok=True)
    staging = target.parent / ('sdar.prepare-' + uuid.uuid4().hex[:10])
    staging.mkdir()
    if local_source:
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'clone', '--no-checkout',
                        '--no-hardlinks', str(local_source.resolve()), str(staging)], check=True)
        git(staging, 'checkout', '--detach', lock['commit'])
    else:
        git(staging, 'init')
        git(staging, 'remote', 'add', 'origin', lock['repository'])
        git(staging, 'fetch', '--depth', '1', 'origin', lock['commit'])
        git(staging, 'checkout', '--detach', 'FETCH_HEAD')
    actual = subprocess.check_output(['git', '-C', str(staging), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == lock['commit']
    git(staging, 'apply', '--check', str(patch))
    git(staging, 'apply', str(patch))
    shutil.copyfile(ROOT / 'configs/opsd.yaml', staging / 'verl/trainer/config/rethink_public.yaml')
    (staging / '.rethink_setup.json').write_text(json.dumps(receipt, indent=2) + '\n')
    staging.rename(target)
    print('Pinned SDAR and ALFWorld method patch prepared.')


def import_data(source):
    lock = json.loads((ROOT / 'configs/data.json').read_text())
    source = source.resolve()
    selected = []
    for name, wanted in lock['files'].items():
        # Public data bundle: exactly the same layout as this repository's data/.
        # A repository/release root containing data/ is also accepted.
        candidates = [source / name, source / 'data' / name]
        if name.startswith('route_sft/'):
            candidates.append(source / 'sft/data_v1' / Path(name).name)  # older bundle compatibility
        existing = [p for p in candidates if p.is_file()]
        if not existing:
            raise RuntimeError('Missing frozen data file: ' + name)
        original = existing[0]
        if sha(original) != wanted:
            raise RuntimeError('Missing or mismatched frozen data file: ' + name)
        dest = ROOT / 'data' / name
        if dest.exists() and sha(dest) != wanted:
            raise RuntimeError('Existing data differs; refusing overwrite: ' + name)
        selected.append((original, dest, wanted))
    for original, dest, wanted in selected:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            pending = dest.with_name(dest.name + '.pending')
            if pending.exists():
                raise RuntimeError('An incomplete copy exists; inspect it before retrying')
            shutil.copyfile(original, pending)
            assert sha(pending) == wanted
            pending.rename(dest)
    print('Frozen dataset copied and SHA-256 verified.')


def bundled_data():
    """Extract only hash-pinned members, then reuse the no-overwrite importer."""
    bundle = json.loads((ROOT / 'configs/data_bundle.json').read_text(encoding='utf-8'))
    lock = json.loads((ROOT / 'configs/data.json').read_text(encoding='utf-8'))
    archive = ROOT / bundle['path']
    if archive.stat().st_size != bundle['bytes'] or sha(archive) != bundle['sha256']:
        raise RuntimeError('Bundled data archive differs from the pinned release')
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(prefix='rethink-data-') as directory:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(lock['files']):
            raise RuntimeError('Unexpected or duplicate data archive members')
        staging = Path(directory).resolve()
        for name in names:
            dest = (staging / name).resolve()
            if not dest.is_relative_to(staging) or z.getinfo(name).is_dir():
                raise RuntimeError('Unsafe data archive member')
            if z.getinfo(name).file_size > 128 * 1024**2:
                raise RuntimeError('Unexpected data archive member size')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, dest.open('xb') as output:
                shutil.copyfileobj(source, output)
            if sha(dest) != lock['files'][name]:
                raise RuntimeError('Bundled data member hash mismatch: ' + name)
        import_data(staging)


def download_base(accepted):
    if not accepted:
        raise RuntimeError('Read the model license and pass --accept-model-license')
    from huggingface_hub import snapshot_download
    config = json.loads((ROOT / 'configs/sft.json').read_text())
    dest = ROOT / config['parent_dir']
    snapshot_download(config['base_model'], revision=config['base_revision'], local_dir=dest,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE', 'README.md'])
    files = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(dest.iterdir()) if p.is_file() and p.name != 'parent_download_receipt.json']
    (dest / 'parent_download_receipt.json').write_text(json.dumps({
        'model_id': config['base_model'], 'revision': config['base_revision'], 'files': files}, indent=2) + '\n')
    print('Pinned base model downloaded and hashed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework', action='store_true')
    p.add_argument('--sdar-source', type=Path, help='Optional existing SDAR Git clone, read-only')
    p.add_argument('--data-source', type=Path, help='Directory of the separately supplied frozen release')
    p.add_argument('--bundled-data', action='store_true', help='Verify and unpack the included frozen dataset')
    p.add_argument('--download-base', action='store_true')
    p.add_argument('--accept-model-license', action='store_true')
    a = p.parse_args()
    if not (a.framework or a.data_source or a.bundled_data or a.download_base):
        p.error('Choose --framework, --bundled-data, --data-source or --download-base')
    if a.data_source and a.bundled_data:
        p.error('Choose either --bundled-data or --data-source')
    if a.sdar_source and not a.framework:
        p.error('--sdar-source requires --framework')
    if a.framework:
        framework(a.sdar_source)
    if a.data_source:
        import_data(a.data_source)
    if a.bundled_data:
        bundled_data()
    if a.download_base:
        download_base(a.accept_model_license)


if __name__ == '__main__':
    main()

