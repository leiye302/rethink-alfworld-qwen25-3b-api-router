"""Audit/export only public source files; never archive the working directory wholesale."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ROOT_FILES = (
    'README.md', 'project.yaml', 'TODO.md', 'LICENSE', '.gitignore',
    '.env.example', 'data/README.md', 'data/frozen_data.zip',
    'checkpoints/.gitkeep', 'src/.gitattributes',
    'results/figures/.gitkeep', 'results/tables/.gitkeep', 'results/demos/.gitkeep',
)
DOCS = ('idea.md', 'related_work.md', 'experiment_log.md')
TREE_SUFFIXES = {
    'src': {'.py', '.txt', '.patch'},
    'scripts': {'.sh'},
    'configs': {'.json', '.yaml', '.txt'},
    '.github': {'.md', '.yml', '.yaml'},
}
# Report locations/categories only, never matched content. These checks are an
# additional safeguard, not a proof that arbitrary new material is anonymized.
PRIVATE_PATTERNS = {
    'credential': re.compile(r'\b(?:oc_sk_|sk-|hf_|ghp_|github_pat_)[A-Za-z0-9_-]{20,}'),
    'private-host': re.compile(r'\bconnect\.[a-z0-9.-]+\.(?:com|cn)\b', re.I),
    'windows-path': re.compile(r'(?<![A-Za-z])(?:[A-Za-z]:[\\/])'),
    'personal-home': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    'private-storage': re.compile(r'/(?:root/autodl-tmp|mnt/[a-z][a-z0-9_-]*)/', re.I),
    'ssh-login': re.compile(r'\bssh\s+-p\s+\d+\s+[^\s]+@'),
}


def public_paths(root=ROOT):
    root = Path(root).resolve()
    names = set(ROOT_FILES) | {'docs/' + name for name in DOCS}
    for directory, suffixes in TREE_SUFFIXES.items():
        for path in (root / directory).rglob('*'):
            relative = path.relative_to(root)
            if any(part.startswith('.') or part == '__pycache__' for part in relative.parts[1:]):
                continue
            if path.is_file() and (path.suffix in suffixes or path.name == 'LICENSE'):
                names.add(relative.as_posix())
    result = []
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            raise RuntimeError('Missing public file: ' + name)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError('Public file points outside repository: ' + name)
        result.append(path)
    return result


def privacy_findings(root=ROOT):
    root = Path(root).resolve()
    found = []

    def scan(name, payload):
        text = payload.decode('utf-8', errors='strict')
        for number, line in enumerate(text.splitlines(), 1):
            for category, pattern in PRIVATE_PATTERNS.items():
                if pattern.search(line):
                    found.append((name, number, category))

    for path in public_paths(root):
        name = path.relative_to(root).as_posix()
        if name == 'data/frozen_data.zip':
            lock = json.loads((root / 'configs/data.json').read_text(encoding='utf-8'))
            bundle = json.loads((root / 'configs/data_bundle.json').read_text(encoding='utf-8'))
            payload = path.read_bytes()
            if len(payload) != bundle['bytes'] or hashlib.sha256(payload).hexdigest() != bundle['sha256']:
                raise RuntimeError('Public data archive hash/size mismatch')
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(lock['files']):
                    raise RuntimeError('Unexpected public archive members')
                for member in names:
                    info = archive.getinfo(member)
                    if info.file_size > 128 * 1024**2:
                        raise RuntimeError('Oversized public data member')
                    data = archive.read(member)
                    if hashlib.sha256(data).hexdigest() != lock['files'][member]:
                        raise RuntimeError('Public data member hash mismatch')
                    if member.endswith('.gz'):
                        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                            data = stream.read(256 * 1024**2 + 1)
                        if len(data) > 256 * 1024**2:
                            raise RuntimeError('Oversized decompressed data member')
                    scan(name + '!' + member, data)
        else:
            scan(name, path.read_bytes())
    return found


def export(destination, root=ROOT):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    findings = privacy_findings(root)
    if findings:
        raise RuntimeError('Privacy check failed; run src/tools/package.py for redacted locations')
    files = public_paths(root)
    if destination.suffix.lower() != '.zip':
        raise ValueError('Use a .zip destination')
    # Exclusive creation: never replace a previous release or copy local secrets.
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100644 << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            data = path.read_bytes()
            if path.suffix not in {'.png', '.zip', '.gz'}:
                data = data.replace(b'\r\n', b'\n')
            archive.writestr(info, data)
    return len(files)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, help='Create a NEW public-only ZIP; omit for audit only')
    args = p.parse_args()
    findings = privacy_findings()
    if findings:
        for name, line, category in findings:
            print(f'{name}:{line}: {category} (content withheld)')
        raise SystemExit(1)
    count = export(args.out) if args.out else len(public_paths())
    print(f'Public-file check passed: {count} files; no private assets selected.')


if __name__ == '__main__':
    main()
