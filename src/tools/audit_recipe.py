"""Audit named ablation deltas against the immutable public 3B recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from ablation import contract

SFT_ALLOWED={'base_model','base_revision','parent_dir','data_dir','output_dir',
    'epochs','total_steps','scheduler_total_steps','full_resume_checkpoint_steps',
    'prepared_world_size','minimum_host_memory_gib','supervision_mode'}
OPSD_ALLOWED={'trainer.n_gpus_per_node','trainer.experiment_name','ray_init.num_gpus',
    'ray_init.address','ray_init.include_dashboard'}
NATIVE_ROUTE_FILES={'verl/trainer/ppo/rethink_local_hints.py',
                    'agent_system/multi_turn_rollout/rollout_loop.py'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def ported_digest(path):
    payload=path.read_bytes().replace(b'\r\n',b'\n')
    if path.name=='build_dataset.py':
        payload=payload.replace(b'ROOT = Path(__file__).resolve().parents[2]',
                                b'ROOT = Path(__file__).resolve().parents[1]')
        payload=payload.replace(b"default=ROOT/'data/route_sft'",b"default=ROOT/'sft/data_v1'")
    return hashlib.sha256(payload).hexdigest()


def flat(value,prefix=''):
    result={}
    for key,item in value.items():
        if isinstance(item,dict):result.update(flat(item,prefix+key+'.'))
        else:result[prefix+key]=item
    return result


def delta(old,new):
    return {k:{'reference':old.get(k),'current':new.get(k)} for k in sorted(set(old)|set(new))
            if old.get(k)!=new.get(k)}


def audit(check_framework=False):
    import yaml
    spec=contract();external=spec['variant']=='external_api_router'
    reference=json.loads((ROOT/'configs/reference_3b.json').read_text())
    assert reference['source_data_verified']
    sft=json.loads((ROOT/'configs/sft.json').read_text())
    opsd=yaml.safe_load((ROOT/'configs/opsd.yaml').read_text())
    sc=delta(reference['sft'],sft);oc=delta(flat(reference['opsd']),flat(opsd))
    allowed=set(OPSD_ALLOWED)
    if external:allowed|={'algorithm.external_router.'+key for key in opsd['algorithm']['external_router']}
    assert set(sc)<=SFT_ALLOWED,'Unexpected SFT change: '+repr(set(sc)-SFT_ALLOWED)
    assert set(oc)<=allowed,'Unexpected OPSD change: '+repr(set(oc)-allowed)
    assert (sft['base_model'],sft['base_revision'])==(spec['base_model'],spec['base_revision'])
    assert (sft['epochs'],sft['total_steps'],sft['prepared_world_size'])==(1,2000,1)
    assert sft['scheduler_total_steps']==reference['sft']['total_steps']==4000
    assert sft['full_resume_checkpoint_steps']==[2000]
    assert sft['supervision_mode']==('full' if external else 'trigger_only')
    assert opsd['trainer']['n_gpus_per_node']==opsd['ray_init']['num_gpus']==1
    assert opsd['ray_init']['address']=='local'
    assert opsd['trainer']['total_training_steps']==150
    assert opsd['algorithm']['sdar']['gate_beta']==5
    if external:
        router=opsd['algorithm']['external_router']
        assert router==dict(enabled=True,model='glm-5.3-flash',
            run_root='${trainer.default_local_dir}/external_router',concurrency=4,max_new_tokens=384,
            temperature=0.,invalid_reasoning_policy='uncertain',evaluation_enabled=False,api_auth_file=None)
    else:assert not opsd['algorithm'].get('external_router',{}).get('enabled',False)
    allowed_sources=set() if external else {'src/sft/route_bce/core.py','src/sft/route_bce/preflight.py'}
    for name,wanted in reference['source_sha256_lf'].items():
        if name not in allowed_sources:
            assert ported_digest(ROOT/name)==wanted,'Changed shared data/semantic code: '+name
    assert json.loads((ROOT/'configs/data.json').read_text())==reference['frozen_data_lock']
    assert digest(ROOT/'configs/seen140_tasks.json')==reference['evaluation_manifest_sha256_lf']
    evaluation=json.loads((ROOT/'configs/eval.json').read_text())
    assert evaluation['task_manifest']=='seen140_tasks.json'
    evaluation['task_manifest']='tasks_seen140.json'
    assert evaluation==reference['evaluation_protocol']
    lock=json.loads((ROOT/'configs/implementation_lock.json').read_text())
    for name,wanted in lock['source_sha256_lf'].items():
        assert digest(ROOT/name)==wanted,'Ablation implementation drift: '+name
    if check_framework:
        for name,wanted in reference['sdar_sha256_lf'].items():
            if external and name in NATIVE_ROUTE_FILES:continue
            assert digest(ROOT/'.deps/sdar'/name)==wanted,'Shared SDAR code changed: '+name
        for name,wanted in lock['native_sha256_lf'].items():
            assert digest(ROOT/'.deps/sdar'/name)==wanted,'Native routing hook drift: '+name
    return dict(passed=True,variant=spec['variant'],sft_enabled=spec['sft_enabled'],
        sft_changes=sc,opsd_changes=oc,frozen_data_and_shared_math_unchanged=True,
        marker_local_weight_preserved=not external,lr_horizon_preserved=True,
        same_eval_tasks_and_seeds=True,API_at_evaluation=False,sdar_core_checked=check_framework,
        gpu_runtime_verified=False,API_quality_calibrated=False,
        attribution_boundary=('External replacement also removes SFT and changes the RL/reference initialization; '
            'not a pure router-only causal comparison.' if external else
            'Only Bad correction/EOS CE is removed; Good targets, marker coefficients and Stage 2 stay fixed.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework',action='store_true');p.add_argument('--out',type=Path)
    a=p.parse_args();value=audit(a.framework)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
