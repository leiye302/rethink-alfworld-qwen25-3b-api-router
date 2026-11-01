"""Offline checks: one request per episode, hint-only repair, original Actor intact."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/opsd'))
import api_worker
import router_protocol as protocol
import router_worker


def episode():
    return {'task': 'put some apple on table.', 'steps': [
        {'step': 0, 'observation': 'On the desk 1, you see a apple 1.',
         'admissible_actions': ['take apple 1 from desk 1', 'go to table 1', 'look'],
         'reasoning': 'I already hold the apple.', 'reasoning_valid': True,
         'action': 'take apple 1 from desk 1', 'feedback': 'You pick up the apple 1 from the desk 1.'},
        {'step': 1, 'observation': 'You pick up the apple 1 from the desk 1.',
         'admissible_actions': ['go to table 1', 'look'], 'reasoning': 'I should inspect the desk.',
         'reasoning_valid': True, 'action': 'look', 'feedback': 'On the desk 1, you see nothing.'},
        {'step': 2, 'observation': 'On the desk 1, you see nothing.',
         'admissible_actions': ['go to table 1', 'look'], 'reasoning': 'I am unsure.',
         'reasoning_valid': True, 'action': 'look', 'feedback': 'On the desk 1, you see nothing.'},
        {'step': 3, 'observation': 'On the desk 1, you see nothing.',
         'admissible_actions': ['go to table 1', 'look'], 'reasoning': '<action>look</action>',
         'reasoning_valid': False, 'action': 'look', 'feedback': 'On the desk 1, you see nothing.'}]}


def result_body(e=None, bad_command=None):
    e = e or episode()
    data = protocol.payload(e)
    first = data['steps'][0]
    command = bad_command or first['allowed_command_ids'][0]
    return {'routes': {'0': 'bad', '1': 'good', '2': 'uncertain', '3': 'uncertain'},
            'errors': {'0': {'quote': e['steps'][0]['reasoning'], 'reason': 'The apple is on the desk, not yet held.'}},
            'hints': {'0': {'command_id': command, 'skill': 'Pick up the visible apple before placing it.'}}}


def response(body=None):
    return {'choices': [{'message': {'content': json.dumps(body or result_body())}, 'finish_reason': 'stop'}],
            'model': 'mock-model', 'usage': {'prompt_tokens': 20, 'completion_tokens': 12}}


def item(e=None):
    e = protocol.normalize_episode(e or episode())
    return {'key': protocol.digest(e), 'episode': e}


class Attr(dict):
    def __getattr__(self, name):
        return self[name]


def native_router():
    import numpy
    def accept(payload, commands, environment):
        if not payload:
            return None, 'invalid_analyzer_output'
        return 'Reasoning skill: ' + payload['reasoning_skill'], None
    fake = SimpleNamespace(atomic_json=api_worker.save, accepted_hint=accept)
    path = ROOT / '.deps/sdar/verl/trainer/ppo/rethink_external_router.py'
    spec = importlib.util.spec_from_file_location('native_external_router_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'verl.trainer.ppo.rethink_local_hints': fake}):
        spec.loader.exec_module(module)
    return module


class RouterProtocolTests(unittest.TestCase):
    def test_whole_trajectory_is_authorized_but_arbitrary_private_metadata_is_not_sent(self):
        e = episode()
        e.update(private_path='DO_NOT_SEND_PRIVATE_METADATA', old_labels='DO_NOT_SEND_OLD_LABELS')
        body = protocol.request_body(e, 'mock-model')
        data = json.loads(body['messages'][1]['content'])
        self.assertEqual(len(data['steps']), 4)
        self.assertEqual(data['steps'][0]['executed_action'], e['steps'][0]['action'])
        self.assertEqual(data['steps'][-1]['feedback_after_action'], e['steps'][-1]['feedback'])
        self.assertNotIn('DO_NOT_SEND', json.dumps(body))
        self.assertEqual((body['temperature'], body['max_tokens']), (0, 1024 + 224 * 4))
        self.assertEqual(protocol.normalize_episode(e), episode() | {'benchmark': 'alfworld'})

    def test_exact_command_dictionary_and_causal_state_ledger(self):
        e = episode()
        data = protocol.payload(e)
        self.assertIsNone(data['steps'][0]['state_before']['carrying'])
        self.assertEqual(data['steps'][1]['state_before']['carrying'], 'apple 1')
        for raw, step in zip(e['steps'], data['steps']):
            self.assertEqual([data['command_dictionary'][i] for i in step['allowed_command_ids']],
                             raw['admissible_actions'])
        e['steps'][0]['feedback'] = 'You pick up the fake 9 from the imaginary 1.'
        changed = protocol.payload(e)
        self.assertEqual(data['steps'][0]['state_before'], changed['steps'][0]['state_before'])
        self.assertNotEqual(data['steps'][1]['state_before'], changed['steps'][1]['state_before'])

    def test_bad_requires_own_reasoning_quote_and_complete_coverage(self):
        e = episode()
        body = result_body(e)
        good = protocol.parse_result(json.dumps(body), e, 'stop')
        self.assertEqual(good['0']['route'], 'bad')
        self.assertIsNotNone(good['0']['payload'])
        body['errors']['0']['quote'] = e['steps'][1]['reasoning']
        bad = protocol.parse_result(json.dumps(body), e, 'stop')
        self.assertEqual(bad['0']['route'], 'uncertain')
        self.assertEqual(bad['0']['error'], 'unanchored_bad_label')
        del body['routes']['3']
        self.assertTrue(all(r['route'] == 'uncertain' for r in
                            protocol.parse_result(json.dumps(body), e, 'stop').values()))

    def test_malformed_duplicate_and_truncated_results_are_not_fake_good(self):
        for text, finish in [('{}', 'stop'), ('not JSON', 'stop'), ('[]', 'stop'),
                             ('{"routes":{},"routes":{}}', 'stop'),
                             (json.dumps(result_body()), 'length')]:
            with self.subTest(text=text):
                parsed = protocol.parse_result(text, episode(), finish)
                self.assertEqual(len(parsed), 4)
                self.assertTrue(all(v['route'] == 'uncertain' and v['error'] for v in parsed.values()))

    def test_missing_reasoning_cannot_be_certified_good_or_bad(self):
        e = episode()
        body = result_body(e)
        body['routes']['3'] = 'good'
        parsed = protocol.parse_result(json.dumps(body), e, 'stop')
        self.assertEqual(parsed['3']['route'], 'uncertain')
        self.assertEqual(parsed['3']['error'], 'missing_valid_reasoning')

    def test_bad_hint_is_not_a_reason_to_change_bad_route(self):
        body = result_body(bad_command='c99999')
        parsed = protocol.parse_result(json.dumps(body), episode(), 'stop')
        self.assertEqual(parsed['0']['route'], 'bad')
        self.assertIsNone(parsed['0']['payload'])
        repair = protocol.repair_body(episode(), parsed, 'mock-model')
        self.assertEqual(set(json.loads(repair['messages'][1]['content'])['hint_targets']), {'0'})
        fixed = protocol.parse_repairs(json.dumps({'hints': result_body()['hints']}), episode(), parsed, 'stop')
        self.assertEqual(fixed['0']['route'], 'bad')
        self.assertIsNotNone(fixed['0']['payload'])
        for key in ['1', '2', '3']:
            self.assertEqual(fixed[key], parsed[key])

    def test_repair_cannot_rejudge_or_add_good_hints(self):
        parsed = protocol.parse_result(json.dumps(result_body(bad_command='c99999')), episode(), 'stop')
        for raw in [{'routes': {'0': 'good'}, 'hints': result_body()['hints']},
                    {'hints': {**result_body()['hints'], '1': result_body()['hints']['0']}}]:
            repaired = protocol.parse_repairs(json.dumps(raw), episode(), parsed, 'stop')
            self.assertEqual(repaired['0']['route'], 'bad')
            self.assertIsNone(repaired['0']['payload'])
            self.assertIsNone(repaired['1']['payload'])

    def test_goal_checks_are_hint_only_and_preserve_transformations(self):
        e = episode()
        e['task'] = 'clean some apple and put it in garbagecan.'
        e['steps'][1]['admissible_actions'] += ['go to garbagecan 1', 'go to sinkbasin 1']
        data = protocol.payload(e)
        step = data['steps'][1]
        command_id = next(k for k, c in data['command_dictionary'].items() if c == 'go to garbagecan 1')
        hint, why = protocol.validate_hint({'command_id': command_id, 'skill': 'Dispose of it now.'}, step, data)
        self.assertIsNone(hint)
        self.assertEqual(why, 'final_delivery_before_required_transformation')
        step['state_before']['confirmed_transformations']['apple 1'] = ['clean']
        hint, why = protocol.validate_hint({'command_id': command_id, 'skill': 'Deliver the cleaned apple.'}, step, data)
        self.assertIsNotNone(hint)
        e = episode()
        e['task'] = 'put some apple in drawer.'
        e['steps'][1]['admissible_actions'] += ['move apple 1 to shelf 1']
        data = protocol.payload(e)
        command_id = next(k for k, c in data['command_dictionary'].items() if c == 'move apple 1 to shelf 1')
        self.assertEqual(protocol.validate_hint({'command_id': command_id, 'skill': 'Place it.'},
                         data['steps'][1], data)[1], 'target_move_to_wrong_goal_destination')

    def test_cache_avoids_charge_and_binds_model_endpoint_and_full_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch.object(api_worker, 'request_json', return_value=response()) as call:
                first = router_worker.execute(item(), cache, 'dummy', 'https://api.example', 'm1')
                again = router_worker.execute(item(), cache, 'dummy', 'https://api.example', 'm1')
                router_worker.execute(item(), cache, 'dummy', 'https://api.example', 'm2')
                router_worker.execute(item(), cache, 'dummy', 'https://other.example', 'm1')
                changed = episode()
                changed['steps'][-1]['feedback'] = 'A new later observation.'
                router_worker.execute(item(changed), cache, 'dummy', 'https://api.example', 'm1')
            self.assertEqual(call.call_count, 4)
            self.assertEqual(first['api_calls_this_run'], 1)
            self.assertEqual(again['api_calls_this_run'], 0)
            self.assertEqual(again['cache_hits'], 1)
            self.assertEqual(again['usage'], [])

    def test_only_failed_hints_are_repaired_once_and_successful_calls_are_cached(self):
        invalid = response(result_body(bad_command='c99999'))
        fixed = response({'hints': result_body()['hints']})
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch.object(api_worker, 'request_json', side_effect=[invalid, fixed]) as call:
                value = router_worker.execute(item(), cache, 'dummy', 'https://api.example', 'm')
            self.assertEqual(call.call_count, 2)
            self.assertEqual(value['repair_targets'], ['0'])
            self.assertIsNotNone(value['steps']['0']['payload'])
            self.assertEqual(value['steps']['1']['route'], 'good')
            with patch.object(api_worker, 'request_json', side_effect=AssertionError('No network')):
                cached = router_worker.execute(item(), cache, 'dummy', 'https://api.example', 'm')
            self.assertEqual(cached['api_calls_this_run'], 0)

    def test_transient_failure_keeps_completed_joint_call_for_retry(self):
        invalid = response(result_body(bad_command='c99999'))
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch.object(api_worker, 'request_json', side_effect=[invalid, RuntimeError('API_TRANSIENT_HOLD')]):
                with self.assertRaisesRegex(RuntimeError, 'API_TRANSIENT_HOLD'):
                    router_worker.execute(item(), cache, 'dummy', 'https://api.example', 'm')
            self.assertEqual(len(list(cache.glob('*.json'))), 1)
            with patch.object(api_worker, 'request_json', return_value=response({'hints': result_body()['hints']})) as call:
                value = router_worker.execute(item(), cache, 'dummy', 'https://api.example', 'm')
            self.assertEqual(call.call_count, 1)
            self.assertEqual(value['api_calls_this_run'], 1)
            self.assertEqual(value['cache_hits'], 1)

    def test_auth_failures_and_corrupt_cache_do_not_become_good(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            for code in ['API_AUTH_OR_REQUEST_HOLD', 'API_TRANSIENT_HOLD']:
                with patch.object(api_worker, 'request_json', side_effect=RuntimeError(code)), self.assertRaisesRegex(RuntimeError, code):
                    router_worker.execute(item(), cache, 'dummy', 'https://api.example', 'm')
            self.assertFalse(list(cache.glob('*.json')))
            with patch.object(api_worker, 'request_json', return_value=response()):
                router_worker.execute(item(), cache, 'dummy', 'https://api.example', 'm')
            file = next(cache.glob('*.json'))
            value = json.loads(file.read_text())
            value['response']['content'] = '{"unexpected":"corruption"}'
            api_worker.save(file, value)
            with self.assertRaisesRegex(RuntimeError, 'ROUTER_CACHE_INTEGRITY_FAILURE'):
                router_worker.execute(item(), cache, 'dummy', 'https://api.example', 'm')

    def test_transport_auth_headers_are_stable_private_and_validate_control_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'auth.json'
            api_worker.save(file, {'token': 'dummy', 'session': 'unit-test-session', 'org_id': 'unit-test-org'})
            file.chmod(0o600)
            with patch.dict(api_worker.AUTH_HEADERS, {}, clear=True):
                self.assertEqual(api_worker.read_auth(file), 'dummy')
                self.assertEqual(api_worker.AUTH_HEADERS['x-opencode-session'], 'unit-test-session')
                self.assertEqual(api_worker.AUTH_HEADERS['x-opencode-org-id'], 'unit-test-org')
                api_worker.save(file, {'token': 'dummy', 'session': 'invalid\r\nheader'})
                # Atomic replacement creates a fresh inode. Restore the private
                # fixture permissions before testing header validation on Linux.
                file.chmod(0o600)
                with self.assertRaisesRegex(RuntimeError, 'INVALID_API_HEADER'):
                    api_worker.read_auth(file)
                if os.name != 'nt':
                    file.chmod(0o644)
                    with self.assertRaisesRegex(RuntimeError, 'AUTH_PERMISSIONS_TOO_OPEN'):
                        api_worker.read_auth(file)


@unittest.skipUnless(importlib.util.find_spec('torch') and (ROOT / '.deps/sdar/.rethink_setup.json').is_file(),
                     'Torch CPU and prepared public source required')
class NativeRouterTests(unittest.TestCase):
    def batch(self):
        import numpy as np
        import torch
        e = episode()
        strings = {21: '<think>' + e['steps'][0]['reasoning'] + '</think>',
                   22: '<action>take apple 1 from desk 1</action>',
                   23: '<think>' + e['steps'][1]['reasoning'] + '</think>',
                   24: '<think>' + e['steps'][2]['reasoning'] + '</think>',
                   25: '<action>look</action>', 26: '<action>look</action>'}
        tok = SimpleNamespace(decode=lambda ids, **_: ''.join(strings.get(i, '') for i in ids))
        batch = SimpleNamespace(batch={'prompts': torch.tensor([[0, 11, 12]] * 4),
            'responses': torch.tensor([[21, 22, 0], [23, 22, 0], [24, 25, 0], [26, 25, 0]]),
            'attention_mask': torch.tensor([[0, 1, 1, 1, 1, 0]] * 4)},
            non_tensor_batch={'rh_actor_rethink': np.array([False, True, False, True]),
                              'rh_episode': np.array([json.dumps(e)] * 4, dtype=object),
                              'rh_step': np.arange(4)})
        return batch, tok

    def test_episode_grouping_uses_one_request_and_exact_step_alignment(self):
        native = native_router()
        batch, tok = self.batch()
        requests, keys = native.episode_requests(batch, tok, protocol)
        self.assertEqual(len(requests), 1)
        self.assertEqual([key[1] if key else None for key in keys], ['0', '1', '2', None])
        self.assertEqual(len(next(iter(requests.values()))['episode']['steps']), 4)
        self.assertTrue(all(key[0] == keys[0][0] for key in keys if key))
        changed = deepcopy(batch)
        value = json.loads(changed.non_tensor_batch['rh_episode'][0])
        value['steps'][0]['reasoning'] = 'Misaligned recorded reasoning.'
        changed.non_tensor_batch['rh_episode'][0] = json.dumps(value)
        with self.assertRaisesRegex(RuntimeError, 'reasoning differ'):
            native.episode_requests(changed, tok, protocol)

    def test_inactive_padding_does_not_need_episode_metadata_or_api(self):
        import torch
        native = native_router()
        batch, tok = self.batch()
        batch.batch['active_masks'] = torch.tensor([[True], [True], [True], [False]])
        batch.non_tensor_batch['rh_episode'][3] = None
        requests, keys = native.episode_requests(batch, tok, protocol)
        self.assertEqual(len(requests), 1)
        self.assertIsNone(keys[3])
        batch.batch['active_masks'].zero_()
        requests, keys = native.episode_requests(batch, tok, protocol)
        self.assertEqual(requests, {})
        self.assertEqual(keys, [None] * 4)

    def test_unregistered_base_marker_only_allowed_for_external_capture(self):
        import torch
        batch, tok = self.batch()
        tok.encode = lambda *_, **__: [8, 9]
        fake_hint = SimpleNamespace(VERSION='test')
        fake_bench = SimpleNamespace(benchmark=lambda x: x, command_allowed=lambda *x: True, hint_command=lambda x: x)
        path = ROOT / '.deps/sdar/verl/trainer/ppo/rethink_local_hints.py'
        spec = importlib.util.spec_from_file_location('native_capture_test', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'verl.trainer.ppo.rethink_hint_protocol': fake_hint,
                                     'verl.trainer.ppo.rethink_benchmark_contract': fake_bench}):
            spec.loader.exec_module(module)
        before = (['A desk.'] * 4, [['look']] * 4)
        envs = SimpleNamespace(memory=[[{'action': 'look'}] for _ in range(4)])
        original = batch.batch['responses'].clone()
        with self.assertRaisesRegex(ValueError, 'registered'):
            module.record_step(batch, tok, before, {'anchor': ['A desk.'] * 4}, [{}] * 4, envs)
        module.record_step(batch, tok, before, {'anchor': ['A desk.'] * 4}, [{}] * 4, envs,
                           allow_unregistered_marker=True)
        torch.testing.assert_close(batch.batch['responses'], original, rtol=0, atol=0)
        self.assertEqual(len(batch.non_tensor_batch['rh_step_record']), 4)

    def run_selection(self, transient=False):
        import numpy as np
        import torch
        native = native_router()
        batch, tok = self.batch()
        tensors = {k: v.clone() for k, v in batch.batch.items()}
        metadata = {k: v.copy() for k, v in batch.non_tensor_batch.items()}
        with tempfile.TemporaryDirectory() as directory:
            auth = Path(directory) / 'auth.json'
            api_worker.save(auth, {'token': 'dummy', 'session': 'unit-test-session'})
            auth.chmod(0o600)
            cfg = Attr(enabled=True, model='mock-model', run_root=directory, concurrency=4,
                       protocol=protocol.VERSION, granularity='full_episode', max_new_tokens=14000,
                       tokens_per_step=224, base_output_tokens=1024, temperature=0.,
                       evaluation_enabled=False, api_auth_file=str(auth))
            trainer = SimpleNamespace(config=SimpleNamespace(algorithm=SimpleNamespace(external_router=cfg)),
                                      tokenizer=tok, global_steps=1)
            calls = []
            def execute(command, **_):
                calls.append(command)
                if transient and len(calls) == 1:
                    raise native.subprocess.CalledProcessError(75, command)
                with patch.object(sys, 'argv', command[1:]):
                    router_worker.main()
            env = {'RETHINK_ROUTER_WORKER': str(ROOT / 'src/opsd/router_worker.py'),
                   'RETHINK_API_ENDPOINT': 'https://api.example'}
            with patch.dict(os.environ, env), patch.object(native.subprocess, 'run', side_effect=execute), \
                 patch.object(api_worker, 'request_json', return_value=response()) as api, \
                 patch.object(native.time, 'sleep') as sleep:
                hints = native.external_hints(trainer, batch, lambda *_: self.fail('No duplicate native API call'))
            self.assertIsNotNone(hints[0])
            self.assertTrue(all(v is None for v in hints[1:]))
            self.assertEqual(api.call_count, 1)
            self.assertEqual(len(calls), 2 if transient else 1)
            if transient:
                sleep.assert_called_once_with(60)
            report = json.loads((Path(directory) / 'reports/router_usage_000001.json').read_text())
            self.assertEqual(report['route_counts'], {'bad': 1, 'good': 1, 'uncertain': 2})
            self.assertEqual(report['episode_requests'], 1)
            self.assertEqual(report['actor_marker_rows'], 2)
            self.assertEqual(report['external_bad_rows'], 1)
            self.assertTrue(report['judge_saw_future_or_episode'])
            self.assertFalse(report['Actor_responses_modified'])
            for key, value in tensors.items():
                torch.testing.assert_close(batch.batch[key], value, rtol=0, atol=0)
            for key, value in metadata.items():
                np.testing.assert_array_equal(batch.non_tensor_batch[key], value)

    def test_joint_selection_is_teacher_only_original_actor_batch_intact(self):
        self.run_selection()

    def test_outage_retains_same_batch_and_reuses_cache(self):
        self.run_selection(True)


if __name__ == '__main__':
    unittest.main()
