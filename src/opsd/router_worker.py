"""One joint request per unique episode; repair rejected hints in one optional call."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import api_worker
import router_protocol as protocol


def source_identity():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ['router_worker.py', 'router_protocol.py', 'api_worker.py']}


def cached_call(body, cache, token, endpoint, operation):
    key = protocol.digest({'body': body, 'endpoint': endpoint, 'protocol': protocol.VERSION,
                           'operation': operation, 'source_sha256': source_identity()})
    path = cache / (key + '.json')
    if path.exists():
        value = json.loads(path.read_text(encoding='utf-8'))
        if (value.get('request_sha256') != key or value.get('schema') != protocol.VERSION or
                value.get('response_sha256') != protocol.digest(value.get('response'))):
            raise RuntimeError('ROUTER_CACHE_INTEGRITY_FAILURE')
        return value['response'], False
    result = api_worker.request_json(body, token, endpoint)
    try:
        choice = result['choices'][0]
        response = {'content': choice['message']['content'], 'finish_reason': choice.get('finish_reason'),
                    'response_model': result.get('model'), 'usage': result.get('usage', {})}
        if not isinstance(response['usage'], dict):
            raise TypeError('Invalid usage')
    except (KeyError, IndexError, TypeError):
        raise RuntimeError('API_MALFORMED_RESPONSE_HOLD') from None
    api_worker.save(path, {'request_sha256': key, 'schema': protocol.VERSION,
                          'response': response, 'response_sha256': protocol.digest(response)})
    return response, True


def execute(item, cache, token, endpoint, model):
    episode = protocol.normalize_episode(item['episode'])
    if item['key'] != protocol.digest(episode):
        raise RuntimeError('ROUTER_UNBOUND_EPISODE')
    responses = []
    primary, called = cached_call(protocol.request_body(episode, model), cache, token, endpoint, 'joint')
    responses.append((primary, called))
    outputs = protocol.parse_result(primary['content'], episode, primary['finish_reason'])
    repair_targets = [k for k, v in outputs.items() if v['route'] == 'bad' and v['payload'] is None]
    if repair_targets:
        repair, called = cached_call(protocol.repair_body(episode, outputs, model),
                                     cache, token, endpoint, 'repair_hints_only')
        responses.append((repair, called))
        outputs = protocol.parse_repairs(repair['content'], episode, outputs, repair['finish_reason'])
    return {'steps': outputs, 'repair_targets': repair_targets,
            'api_calls_this_run': sum(called for _, called in responses),
            'cache_hits': sum(not called for _, called in responses),
            'usage': [value['usage'] for value, called in responses if called],
            'response_models': [value['response_model'] for value, _ in responses]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--auth', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    endpoint = api_worker.validate_endpoint(os.environ.get('RETHINK_API_ENDPOINT', ''))
    token = api_worker.read_auth(args.auth)
    payload = json.loads(args.request.read_text(encoding='utf-8'))
    if payload.get('schema') != protocol.VERSION:
        raise RuntimeError('ROUTER_PROTOCOL_MISMATCH')
    if protocol.digest({k: v for k, v in payload.items() if k != 'request_sha256'}) != payload['request_sha256']:
        raise RuntimeError('ROUTER_REQUEST_INTEGRITY_FAILURE')
    if payload.get('endpoint_sha256') != protocol.digest(endpoint):
        raise RuntimeError('ROUTER_ENDPOINT_IDENTITY_FAILURE')
    if payload.get('source_sha256') != source_identity():
        raise RuntimeError('ROUTER_SOURCE_IDENTITY_FAILURE')
    keys = [item['key'] for item in payload['requests']]
    if (len(keys) != len(set(keys)) or any(item['key'] != protocol.digest(protocol.normalize_episode(item['episode']))
                                         for item in payload['requests'])):
        raise RuntimeError('ROUTER_DUPLICATE_OR_UNBOUND_EPISODE')
    workers = int(payload['concurrency'])
    if not 1 <= workers <= 4:
        raise ValueError('Use 1 to 4 episode workers')
    args.cache.mkdir(parents=True, exist_ok=True)
    outputs = {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, item, args.cache, token, endpoint, payload['model']): item['key']
                   for item in payload['requests']}
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
    usage = [u for value in outputs.values() for u in value['usage']]
    all_costs = [u['cost'] for u in usage if isinstance(u.get('cost'), (int, float))]
    allowances = [u['opencode']['allowance'] for u in usage
                  if isinstance(u.get('opencode'), dict) and isinstance(u['opencode'].get('allowance'), (int, float))]
    api_worker.save(args.output, {'request_sha256': payload['request_sha256'], 'schema': protocol.VERSION,
        'outputs': outputs, 'episode_requests': len(outputs),
        'api_calls_this_run': sum(v['api_calls_this_run'] for v in outputs.values()),
        'cache_hits': sum(v['cache_hits'] for v in outputs.values()),
        'repair_episode_requests': sum(bool(v['repair_targets']) for v in outputs.values()),
        'prompt_tokens': sum(u.get('prompt_tokens', 0) for u in usage),
        'completion_tokens': sum(u.get('completion_tokens', 0) for u in usage),
        'provider_reported_cost': sum(all_costs) if len(all_costs) == len(usage) and usage else None,
        'Go_allowance_USD': sum(allowances) if allowances else None,
        'seconds': time.monotonic() - started,
        'judge_saw_current_action': True, 'judge_saw_future_or_episode': True,
        'student_modified': False, 'student_sees_hint': False})


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        code = str(exc)
        print(code if code.isupper() and code.replace('_', '').isalnum() else 'ROUTER_WORKER_FAILURE')
        raise SystemExit(75 if code == 'API_TRANSIENT_HOLD' else
                         76 if code in {'API_AUTH_OR_REQUEST_HOLD', 'API_MALFORMED_RESPONSE_HOLD'} else 77)
    except Exception:
        print('ROUTER_WORKER_INVALID_INPUT_OR_CACHE')
        raise SystemExit(77)
