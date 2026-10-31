"""CPU-only packaging checks; --assets also validates local training prerequisites."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import os
import sys

from package import privacy_findings


def check_runtime(stage):
    """Import the actual entry-point dependencies; do not initialize Ray/CUDA."""
    import importlib
    import importlib.metadata
    import re
    requirements = ROOT / f'configs/requirements-{stage}.txt'
    for raw in requirements.read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([\w-]+)(?:\[[\w,]+\])?==([\w.]+)', raw.strip())
        # CUDA wheels carry local version suffixes, e.g. +cu128; PEP 440 ==
        # accepts that local build when the requested version has no suffix.
        if match and importlib.metadata.version(match[1]).split('+', 1)[0] != match[2]:
            raise RuntimeError('Runtime version mismatch: ' + match[1])
    if stage == 'sft':
        sys.path.insert(0, str(ROOT / 'src/sft'))
        modules = ('route_bce.preflight', 'route_bce.train')
    else:
        check_framework()
        sys.path.insert(0, str(ROOT / '.deps/sdar'))
        modules = ('flash_attn', 'vllm', 'verl.trainer.main_sdar', 'verl.workers.fsdp_workers',
                   'agent_system.environments.env_package.alfworld')
    for name in modules:
        importlib.import_module(name)
    print(stage + ' runtime versions/imports passed; no training or API request performed.')


def check_framework():
    from setup import verify_framework
    upstream = json.loads((ROOT / 'configs/upstream.json').read_text(encoding='utf-8'))
    receipt = dict(upstream, config_sha256=sha(ROOT / 'configs/opsd.yaml'))
    verify_framework(ROOT / '.deps/sdar', upstream, ROOT / 'src/opsd/sdar.patch', receipt)

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', action='store_true')
    p.add_argument('--runtime', choices=['sft', 'opsd'], help='Check installed versions and import runtime entry points')
    a = p.parse_args()
    findings = privacy_findings()
    assert not findings, 'Public-file privacy check failed; run src/tools/package.py for redacted locations'
    for directory in ('src', 'scripts'):
        for path in (ROOT / directory).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.name)
    cfg = json.loads((ROOT / 'configs/sft.json').read_text())
    assert (cfg['route_lambda'], cfg['reference_kl_coef']) == (.1, 0)
    assert (cfg['good_per_batch'], cfg['bad_per_batch'], cfg['total_steps']) == (5, 3, 2000)
    assert (cfg['epochs'], cfg['prepared_world_size'], cfg['scheduler_total_steps']) == (1, 1, 4000)
    sys.path.insert(0,str(ROOT/'src'))
    from ablation import contract
    spec=contract()
    assert cfg['supervision_mode']==('trigger_only' if spec['sft_enabled'] else 'full')
    from audit_recipe import audit
    assert audit()['passed']
    tasks = json.loads((ROOT / 'configs/seen140_tasks.json').read_text())
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 140
    assert all('/valid_seen/' in t['task_id'] and t['seed'] == 2026+i for i, t in enumerate(tasks))
    upstream = json.loads((ROOT / 'configs/upstream.json').read_text())
    assert sha(ROOT / 'src/opsd/sdar.patch') == upstream['patch_sha256']
    for name in ('train.sh', 'eval.sh', 'reproduce.sh'):
        assert (ROOT / 'scripts' / name).read_text().startswith('#!/usr/bin/env bash')
    if a.assets:
        data = json.loads((ROOT / 'configs/data.json').read_text())
        for name, wanted in data['files'].items():
            assert sha(ROOT / 'data' / name) == wanted, 'Frozen data mismatch: ' + name
        model = ROOT / cfg['parent_dir']
        receipt = json.loads((model / 'parent_download_receipt.json').read_text())
        assert receipt['model_id'] == cfg['base_model'] and receipt['revision'] == cfg['base_revision']
        for item in receipt['files']:
            assert sha(model / item['name']) == item['sha256']
        check_framework()
        assets = Path(os.environ['ALFWORLD_DATA']).resolve()
        for item in tasks:
            game = (assets / item['task_id']).resolve()
            assert game.is_relative_to(assets / 'json_2.1.1/valid_seen')
            assert sha(game) == item['game_sha256']
    if a.runtime:
        check_runtime(a.runtime)
    print('CPU package checks passed; no model generation, training or API request performed.')


if __name__ == '__main__':
    main()

