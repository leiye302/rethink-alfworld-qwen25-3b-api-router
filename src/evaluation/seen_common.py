"""Small CPU-only helpers for the post-training fixed-seen evaluation."""
import hashlib
import json
import os
from pathlib import Path
import re


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def protocol(text, ids, marker_id):
    # Never strip RETHINK. Telemetry uses the raw token IDs as well as text.
    clean = text.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
    think = re.search(r'<think>(.*?)</think>', clean, re.S)
    complete = re.fullmatch(
        r'<think>(.*?)</think>\s*(?:<RETHINK>\s*)?<action>(.*?)</action>', clean, re.S)
    strict = bool(complete and complete[1].strip() and complete[2].strip()
                  and clean.count('<think>') == clean.count('</think>') == 1
                  and clean.count('<action>') == clean.count('</action>') == 1
                  and clean.count('<RETHINK>') <= 1
                  and '<RETHINK>' not in complete[1]
                  and '<RETHINK>' not in complete[2])
    count = ids.count(marker_id) if marker_id is not None else 0
    return {'reasoning_open': '<think>' in clean,
            'reasoning_nonempty_closed': bool(think and think[1].strip()),
            'strict_protocol': strict,
            'marker_token_count': count,
            'marker_text_count': clean.count('<RETHINK>'),
            'marker_triggered': bool(count or '<RETHINK>' in clean)}


def readiness(completion, supervisor, live_training, gpu_busy):
    if not completion:
        return 'waiting_for_training_completion'
    if (completion.get('status') != 'completed' or completion.get('step') != 150
            or not completion.get('full_checkpoint_hash_verified')):
        return 'completion_not_verified'
    if supervisor.get('status') != 'completed' or live_training:
        return 'waiting_for_training_exit'
    if gpu_busy:
        return 'waiting_for_gpu_release'
    return 'ready'


def family(task_id):
    name = Path(task_id).parts[2]
    for prefix in ('pick_two_obj_and_place', 'look_at_obj_in_light',
                   'pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep',
                   'pick_heat_then_place_in_recep', 'pick_and_place_simple'):
        if name.startswith(prefix + '-'):
            return prefix
    raise ValueError('UNKNOWN_TASK_FAMILY')
