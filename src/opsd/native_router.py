"""Native SDAR hook: full-episode critic selects only Teacher-local hints.

This file is also installed by the pinned framework patch. No Actor tensor,
marker flag, environment action, reward, response mask or SDAR loss is replaced.
"""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from verl.trainer.ppo.rethink_local_hints import atomic_json, accepted_hint

VERSION = 'external_hindsight_episode_router_v2'


def load_protocol(worker):
    path = worker.with_name('router_protocol.py')
    spec = importlib.util.spec_from_file_location('external_episode_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.VERSION != VERSION:
        raise RuntimeError('External episode protocol differs from the native hook')
    return module


def response_reasoning(batch, tokenizer, row):
    response = batch.batch['responses'][row]
    mask = batch.batch['attention_mask'][row, -len(response):].bool()
    text = tokenizer.decode(response[mask].tolist(), skip_special_tokens=False)
    matches = list(re.finditer(r'<think>(.*?)</think>', text, re.S))
    if (len(matches) != 1 or text.count('<think>') != 1 or text.count('</think>') != 1 or
            not matches[0][1].strip() or ('<action>' in text and text.find('<action>') < matches[0].end())):
        return None
    return matches[0][1].strip()


def episode_requests(batch, tokenizer, protocol):
    groups, rows, decoded = {}, [], {}
    for row in range(len(batch.batch['responses'])):
        active = batch.batch.get('active_masks')
        if active is not None and not bool(active[row].bool().any()):
            rows.append(None)
            continue
        raw = batch.non_tensor_batch['rh_episode'][row]
        if raw is None and response_reasoning(batch, tokenizer, row) is None:
            rows.append(None)
            continue
        if raw not in decoded:
            episode = protocol.normalize_episode(json.loads(raw))
            decoded[raw] = (episode, protocol.digest(episode))
        episode, key = decoded[raw]
        target = int(batch.non_tensor_batch['rh_step'][row])
        if target < 0 or target >= len(episode['steps']):
            raise RuntimeError('Episode target step is out of bounds')
        group = groups.setdefault(key, {'episode': episode, 'validity': {}})
        reasoning = response_reasoning(batch, tokenizer, row)
        valid = reasoning is not None
        if valid and reasoning != episode['steps'][target]['reasoning']:
            raise RuntimeError('Actor response and recorded episode reasoning differ')
        if target in group['validity'] and group['validity'][target] != valid:
            raise RuntimeError('Conflicting response validity at the same episode step')
        group['validity'][target] = valid
        rows.append((key, target, valid))
    requests, final_keys = {}, {}
    for key, group in groups.items():
        if not any(group['validity'].values()):
            continue
        episode = group['episode']
        for step in episode['steps']:
            # Unknown/unrepresented or malformed reasoning is not certified Good.
            step['reasoning_valid'] = group['validity'].get(step['step'], False)
        final = protocol.digest(episode)
        final_keys[key] = final
        requests.setdefault(final, {'key': final, 'episode': episode})
    row_keys = [(final_keys[row[0]], str(row[1])) if row is not None and row[2] and row[0] in final_keys
                else None for row in rows]
    return requests, row_keys


def external_hints(trainer, batch, native_generate=None):
    # native_generate remains an unused compatibility argument. Calling it here
    # would duplicate API work and could silently rejudge the selected positions.
    cfg = trainer.config.algorithm.external_router
    if cfg.get('evaluation_enabled', False):
        raise ValueError('This ablation forbids evaluation-time API routing')
    if (cfg.get('protocol') != VERSION or cfg.get('granularity') != 'full_episode' or
            cfg.get('temperature') != 0 or cfg.get('max_new_tokens') != 14000 or
            cfg.get('tokens_per_step') != 224 or cfg.get('base_output_tokens') != 1024):
        raise ValueError('Version the episode sampling protocol before changing it')
    worker = Path(os.environ['RETHINK_ROUTER_WORKER']).resolve()
    protocol = load_protocol(worker)
    root = Path(cfg.run_root).resolve()
    sources = {name: hashlib.sha256(worker.with_name(name).read_bytes()).hexdigest()
               for name in ['router_worker.py', 'router_protocol.py', 'api_worker.py']}
    requests, row_keys = episode_requests(batch, trainer.tokenizer, protocol)
    payload = {'schema': VERSION, 'requests': list(requests.values()), 'model': cfg.model,
               'concurrency': int(cfg.concurrency), 'source_sha256': sources,
               'endpoint_sha256': protocol.digest(os.environ.get('RETHINK_API_ENDPOINT', ''))}
    identity = protocol.digest(payload)
    payload['request_sha256'] = identity
    stem = root / 'ipc' / f'update_{trainer.global_steps:06d}_{identity}'
    request_path = stem.with_suffix('.requests.json')
    output_path = stem.with_suffix('.results.json')
    if requests and not output_path.exists():
        atomic_json(request_path, payload)
        command = [sys.executable, str(worker), str(request_path), str(output_path), '--auth',
                   str(cfg.api_auth_file), '--cache', str(root / 'api_cache')]
        with stem.with_suffix('.router.log').open('ab', buffering=0) as log:
            while True:
                try:
                    subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                   check=True, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    code = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 75
                    if code not in {75, 76}:
                        raise RuntimeError('ROUTER_INTEGRITY_FAILURE_SEE_PRIVATE_LOG') from None
                    atomic_json(root / 'api_wait_status.json', {'status': 'waiting_same_batch',
                        'step': trainer.global_steps, 'exit_code': code, 'request_sha256': identity, 'time': time.time()})
                    time.sleep(60 if code == 75 else 300)
        atomic_json(root / 'api_wait_status.json', {'status': 'ready', 'step': trainer.global_steps,
                                                   'request_sha256': identity, 'time': time.time()})
    result = json.loads(output_path.read_text(encoding='utf-8')) if requests else {
        'request_sha256': identity, 'outputs': {}}
    if result['request_sha256'] != identity or set(result['outputs']) != set(requests):
        raise RuntimeError('Router result identity/episode coverage mismatch')
    for key, item in requests.items():
        wanted = {str(s['step']) for s in item['episode']['steps']}
        if set(result['outputs'][key]['steps']) != wanted:
            raise RuntimeError('Router result step coverage mismatch')
    original = np.array(batch.non_tensor_batch['rh_actor_rethink'], dtype=bool, copy=True)
    hints, labels, fallback = [], [], Counter()
    for key in row_keys:
        if key is None:
            labels.append('uncertain')
            hints.append(None)
            fallback['missing_valid_reasoning'] += 1
            continue
        episode_key, step_key = key
        value = result['outputs'][episode_key]['steps'][step_key]
        label = value['route']
        if label not in {'good', 'bad', 'uncertain'}:
            raise RuntimeError('Invalid external route label')
        labels.append(label)
        if label != 'bad':
            if value.get('payload') is not None:
                raise RuntimeError('Local hint leaked into a non-Bad external route')
            hints.append(None)
            if value.get('error'):
                fallback[value['error']] += 1
            continue
        step = requests[episode_key]['episode']['steps'][int(step_key)]
        hint, reason = accepted_hint(value.get('payload'), step['admissible_actions'], 'alfworld')
        hints.append(hint)
        if hint is None:
            fallback[value.get('hint_error') or reason or 'invalid_local_hint'] += 1
    if not np.array_equal(batch.non_tensor_batch['rh_actor_rethink'], original):
        raise RuntimeError('Original Actor route metadata was modified')
    report = {'step': trainer.global_steps, 'rows': len(labels), 'episode_requests': len(requests),
        'route_counts': dict(Counter(labels)), 'actor_marker_rows': int(original.sum()),
        'external_bad_rows': labels.count('bad'), 'teacher_local_hint_rows': sum(h is not None for h in hints),
        'fallback_counts': dict(fallback), 'judge_saw_current_action': True,
        'judge_saw_future_or_episode': True, 'Actor_responses_modified': False, 'student_sees_hint': False,
        'fixed_initial_teacher': False, 'teacher_scoring': 'public_pre_update_current_actor',
        'API_calls_this_run': result.get('api_calls_this_run', 0), 'cache_hits': result.get('cache_hits', 0),
        'repair_episode_requests': result.get('repair_episode_requests', 0),
        'prompt_tokens': result.get('prompt_tokens', 0), 'completion_tokens': result.get('completion_tokens', 0),
        'provider_reported_cost': result.get('provider_reported_cost'),
        'Go_allowance_USD': result.get('Go_allowance_USD'),
        'request_sha256': identity, 'model': cfg.model, 'protocol': VERSION}
    atomic_json(root / 'reports' / f'router_usage_{trainer.global_steps:06d}.json', report)
    return hints
