"""Small shared experiment contract; never reads credentials or starts work."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL='Qwen/Qwen2.5-3B-Instruct'
REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'


def contract():
    value=json.loads((ROOT/'configs/ablation.json').read_text(encoding='utf-8'))
    if value['variant'] not in {'trigger_only','external_api_router'}:
        raise ValueError('Unknown ablation variant')
    if value['base_model']!=MODEL or value['base_revision']!=REVISION:
        raise ValueError('The ablations must use the pinned Qwen2.5-3B base')
    if value['API_at_evaluation'] is not False:
        raise ValueError('Both ablations require API-free evaluation')
    if value['sft_enabled'] != (value['variant']=='trigger_only'):
        raise ValueError('SFT enablement does not match the declared ablation')
    return value


def source_binding():
    paths=list((ROOT/'src/opsd').glob('*.py'))+list((ROOT/'src/sft/route_bce').glob('*.py'))
    paths.extend([ROOT/'src/opsd/sdar.patch',ROOT/'configs/ablation.json'])
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
