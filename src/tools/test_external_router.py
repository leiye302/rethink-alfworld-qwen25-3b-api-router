"""No paid calls: request privacy, cache isolation and Teacher-only selection."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/opsd'))
import api_worker
import router_protocol as protocol
import router_worker


def response(route='bad',quote='I already hold the apple.'):
    return {'choices':[{'message':{'content':json.dumps(dict(route=route,error_quote=quote,
            evidence='The visible observation says the apple is on the desk.'))},'finish_reason':'stop'}],
            'model':'mock-model','usage':{'prompt_tokens':20,'completion_tokens':12}}


class Attr(dict):
    def __getattr__(self,name):return self[name]


def native_router():
    # Import before patch.dict takes its module snapshot. Otherwise restoring
    # sys.modules unloads NumPy while the tested module still holds it, which
    # creates two distinct _NoValue singletons on a later import.
    import numpy
    path=ROOT/'.deps/sdar/verl/trainer/ppo/rethink_external_router.py'
    fake=SimpleNamespace(atomic_json=api_worker.save,digest=protocol.digest)
    spec=importlib.util.spec_from_file_location('native_external_router_test',path)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'verl.trainer.ppo.rethink_local_hints':fake}):spec.loader.exec_module(module)
    return module


class RouterProtocolTests(unittest.TestCase):
    def test_request_serializes_only_H_R_not_current_action_episode_or_outcome(self):
        item=dict(H='Visible task/history/observation',R='I already hold the apple.',
                  action='SECRET_CURRENT_ACTION',episode='SECRET_FUTURE',won='SECRET_OUTCOME')
        body=protocol.request_body(item,'mock-model')
        self.assertEqual(json.loads(body['messages'][1]['content']),{'H':item['H'],'R':item['R']})
        self.assertFalse(any(value in json.dumps(body) for value in ('SECRET_CURRENT_ACTION','SECRET_FUTURE','SECRET_OUTCOME')))
        self.assertEqual((body['temperature'],body['max_tokens']),(0,384))

    def test_bad_requires_anchored_error_and_visible_evidence(self):
        r='I already hold the apple.'
        value=response()['choices'][0]['message']['content']
        self.assertEqual(protocol.parse_result(value,r,'stop')['route'],'bad')
        self.assertEqual(protocol.parse_result(value,r,'length')['route'],'uncertain')
        self.assertEqual(protocol.parse_result(value,'Unrelated reasoning','stop')['route'],'uncertain')
        for invalid in ('not JSON','[]','{}','{"route":"BAD"}',
                        '{"route":"bad","error_quote":"","evidence":"x"}'):
            self.assertEqual(protocol.parse_result(invalid,r,'stop')['route'],'uncertain')
        for route in ('good','uncertain'):
            value=json.dumps(dict(route=route,error_quote='',evidence='No clear unrepaired error'))
            self.assertEqual(protocol.parse_result(value,r,'stop')['route'],route)

    def test_cache_avoids_charge_and_binds_model_endpoint_and_reasoning(self):
        item={'H':'Observation: apple on desk.','R':'I already hold the apple.'}
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)
            with patch.object(api_worker,'request_json',return_value=response()) as call:
                first=router_worker.execute(item,cache,'dummy','https://api.example','m1')
                again=router_worker.execute(item,cache,'dummy','https://api.example','m1')
                router_worker.execute(item,cache,'dummy','https://api.example','m2')
                router_worker.execute(item,cache,'dummy','https://other.example','m1')
                router_worker.execute({**item,'H':'A changed visible state'},cache,'dummy','https://api.example','m1')
            self.assertEqual(call.call_count,4)
            self.assertFalse(first['cache_hit']);self.assertTrue(again['cache_hit'])
            self.assertEqual(again['usage'],{})
            self.assertEqual(again['route'],'bad')

    def test_auth_and_transient_failures_do_not_become_good_labels(self):
        item={'H':'Visible state','R':'A reasoning assertion'}
        with tempfile.TemporaryDirectory() as directory:
            for code in ('API_AUTH_OR_REQUEST_HOLD','API_TRANSIENT_HOLD'):
                with patch.object(api_worker,'request_json',side_effect=RuntimeError(code)),self.assertRaisesRegex(RuntimeError,code):
                    router_worker.execute(item,Path(directory),'dummy','https://api.example','m')
            self.assertFalse(list(Path(directory).glob('*.json')))

    @unittest.skipUnless((ROOT/'.deps/sdar/.rethink_setup.json').is_file(),'Prepare framework first')
    def test_hint_prompt_changes_only_selection_provenance(self):
        original=api_worker.load_protocol()
        step=dict(step=0,observation='A desk.',admissible_actions=['go to desk 1'],
                  reasoning='Inspect desk',action='go to desk 1',feedback='A book.')
        request=dict(key='r',target=0,episode=dict(task='Find book',steps=[step]))
        group,=original.group_requests([request])
        native=original.request_body(group)
        captured=[]
        result={'choices':[{'message':{'content':'{"hints":{"0":"Next command: \'go to desk 1\'."}}'},
                           'finish_reason':'stop'}]}
        def capture(body,*_):captured.append(body);return result
        with tempfile.TemporaryDirectory() as directory,patch.object(api_worker,'request_json',side_effect=capture):
            api_worker.execute({**group,'route_source':'external_api'},Path(directory),'dummy','https://api.example')
        changed=captured[0]
        before='The Actor has ALREADY emitted RETHINK at every requested target step.'
        after='An external reasoning-only judge has selected every requested target step.'
        self.assertEqual(changed['messages'][0]['content'],native['messages'][0]['content'].replace(before,after))
        self.assertEqual(changed['messages'][1:],native['messages'][1:])
        self.assertEqual(changed['model'],native['model'])


@unittest.skipUnless(importlib.util.find_spec('torch') and (ROOT/'.deps/sdar/.rethink_setup.json').is_file(),
                     'Torch CPU and prepared source required; no GPU needed')
class NativeRouterTests(unittest.TestCase):
    def test_unregistered_base_marker_allowed_only_for_external_capture(self):
        import torch
        batch,tok=self.batch()
        tok.encode=lambda *_,**__: [8,9]  # base has no single registered marker
        path=ROOT/'.deps/sdar/verl/trainer/ppo/rethink_local_hints.py'
        spec=importlib.util.spec_from_file_location('native_capture_test',path)
        module=importlib.util.module_from_spec(spec)
        fake_hint=SimpleNamespace(VERSION='test')
        fake_bench=SimpleNamespace(benchmark=lambda x:x,command_allowed=lambda *x:True,hint_command=lambda x:x)
        with patch.dict(sys.modules,{'verl.trainer.ppo.rethink_hint_protocol':fake_hint,
                                     'verl.trainer.ppo.rethink_benchmark_contract':fake_bench}):
            spec.loader.exec_module(module)
        before=(['A desk.']*4,[['look']]*4)
        envs=SimpleNamespace(memory=[[{'action':'look'}] for _ in range(4)])
        next_obs={'anchor':['A desk.']*4};infos=[{'won':False} for _ in range(4)]
        original=batch.batch['responses'].clone()
        with self.assertRaisesRegex(ValueError,'registered'):
            module.record_step(batch,tok,before,next_obs,infos,envs)
        module.record_step(batch,tok,before,next_obs,infos,envs,allow_unregistered_marker=True)
        torch.testing.assert_close(batch.batch['responses'],original,rtol=0,atol=0)
        self.assertEqual(len(batch.non_tensor_batch['rh_step_record']),4)

    def batch(self):
        import numpy as np
        import torch
        strings={11:'Task: obtain the apple. ',12:'Visible: apple on desk.',
                 21:'<think>I already hold the apple.</think>',
                 22:'<action>take apple 1 from desk 1</action>',
                 23:'<think>I should inspect the desk.</think>',
                 24:'<think>I am unsure.</think>',25:'<action>look</action>',
                 26:'<action>look</action>'}
        tok=SimpleNamespace(decode=lambda ids,**_:''.join(strings.get(i,'') for i in ids))
        batch=SimpleNamespace(batch={'prompts':torch.tensor([[0,11,12]]*4),
            'responses':torch.tensor([[21,22,0],[23,22,0],[24,25,0],[26,25,0]]),
            'attention_mask':torch.tensor([[0,1,1,1,1,0]]*4)},
            non_tensor_batch={'rh_actor_rethink':np.array([False,True,False,True])})
        return batch,tok

    def test_actual_masked_prefix_only_and_no_future_action_leak(self):
        native=native_router();batch,tok=self.batch()
        requests,keys=native.actor_visible_pairs(batch,tok)
        self.assertEqual(len(requests),3);self.assertIsNone(keys[3])
        self.assertEqual(requests[keys[0]]['H'],'Task: obtain the apple. Visible: apple on desk.')
        self.assertEqual(requests[keys[0]]['R'],'I already hold the apple.')
        self.assertFalse(any('<action>' in json.dumps(v) for v in requests.values()))
        batch.batch['responses'][1]=batch.batch['responses'][0]
        requests,keys=native.actor_visible_pairs(batch,tok)
        self.assertEqual(keys[0],keys[1])
        self.assertEqual(len(requests),2)

    def run_selection(self,transient=False):
        import numpy as np
        import torch
        native=native_router();batch,tok=self.batch()
        tensors={k:v.clone() for k,v in batch.batch.items()}
        flags=batch.non_tensor_batch['rh_actor_rethink'].copy()
        with tempfile.TemporaryDirectory() as directory:
            cfg=Attr(enabled=True,model='mock-model',run_root=directory,concurrency=4,
                     max_new_tokens=384,temperature=0.,evaluation_enabled=False,api_auth_file='unused')
            trainer=SimpleNamespace(config=SimpleNamespace(algorithm=SimpleNamespace(external_router=cfg)),
                                    tokenizer=tok,global_steps=1)
            def worker(command,**_):
                payload=json.loads(Path(command[2]).read_text())
                values={item['key']:{'route':('bad' if 'already' in item['R'] else
                                           'good' if 'inspect' in item['R'] else 'uncertain')}
                        for item in payload['requests']}
                api_worker.save(Path(command[3]),dict(request_sha256=payload['request_sha256'],outputs=values))
            calls=[]
            def execute(command,**kw):
                calls.append(command)
                if transient and len(calls)==1:raise native.subprocess.CalledProcessError(75,command)
                return worker(command,**kw)
            def hints(_trainer,proxy):
                self.assertIs(proxy.batch,batch.batch)
                self.assertIsNot(proxy.non_tensor_batch,batch.non_tensor_batch)
                np.testing.assert_array_equal(proxy.non_tensor_batch['rh_actor_rethink'],[True,False,False,False])
                np.testing.assert_array_equal(proxy.non_tensor_batch['rh_original_actor_rethink'],flags)
                return ['Reasoning skill: inspect inventory',None,None,None]
            env={'RETHINK_ROUTER_WORKER':str(ROOT/'src/opsd/router_worker.py'),
                 'RETHINK_API_ENDPOINT':'https://api.example'}
            with patch.dict(os.environ,env),patch.object(native.subprocess,'run',side_effect=execute), \
                 patch.object(native.time,'sleep') as sleep:
                result=native.external_hints(trainer,batch,hints)
            self.assertIsNotNone(result[0]);self.assertTrue(all(v is None for v in result[1:]))
            self.assertEqual(len(calls),2 if transient else 1)
            if transient:sleep.assert_called_once_with(60)
            report=json.loads((Path(directory)/'reports/router_usage_000001.json').read_text())
            self.assertEqual(report['route_counts'],{'bad':1,'good':1,'uncertain':2})
            self.assertEqual(report['actor_marker_rows'],2)
            self.assertEqual(report['external_bad_rows'],1)
            self.assertFalse(report['Actor_responses_modified'])
        np.testing.assert_array_equal(batch.non_tensor_batch['rh_actor_rethink'],flags)
        for key,value in tensors.items():torch.testing.assert_close(batch.batch[key],value,rtol=0,atol=0)

    def test_bad_selection_is_teacher_only_original_actor_batch_intact(self):self.run_selection()

    def test_transient_holds_same_batch_then_retries_not_fake_good(self):self.run_selection(transient=True)


if __name__=='__main__':unittest.main()
