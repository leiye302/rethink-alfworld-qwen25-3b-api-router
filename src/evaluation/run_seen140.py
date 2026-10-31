"""Evaluate any exported HF checkpoint with the same native seen140 protocol."""
import argparse, hashlib, json, os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--step',type=int,default=None,help='Optional metadata only')
    a=p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('Explicit GPU allocation required')
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import pandas as pd
    model,assets,out=a.model.resolve(),a.assets.resolve(),a.out.resolve()
    assert list(model.glob('*.safetensors')),'Export FSDP checkpoint to Hugging Face format first'
    tasks=json.loads((ROOT/'configs/seen140_tasks.json').read_text())
    for i,t in enumerate(tasks):
        f=(assets/t['task_id']).resolve()
        assert f.is_relative_to(assets/'json_2.1.1/valid_seen') and sha(f)==t['game_sha256'] and t['seed']==2026+i
    out.mkdir(parents=True,exist_ok=False)
    (out/'tasks_alfworld_seen140.json').write_bytes((ROOT/'configs/seen140_tasks.json').read_bytes())
    records=[{'data_source':'text','prompt':[{'role':'user','content':''}],'ability':'agent',
              'extra_info':{'index':i,'split':'test'}} for i in range(140)]
    pd.DataFrame(records).to_parquet(out/'seen.parquet',index=False)
    with initialize_config_dir(config_dir=str(ROOT/'.deps/sdar/verl/trainer/config'),version_base=None):
        cfg=compose(config_name='rethink_public')
    changes={'actor_rollout_ref.model.path':str(model),'data.train_files':str(out/'seen.parquet'),
        'data.val_files':str(out/'seen.parquet'),'env.alfworld.fixed_eval_manifest':str(out/'tasks_alfworld_seen140.json'),
        'trainer.val_only':True,'trainer.default_local_dir':str(out/'unused'),
        'trainer.validation_data_dir':str(out/'validation'),'trainer.log_val_generations':0,
        'algorithm.local_hints.enabled':False,'algorithm.local_hints.api_auth_file':'unused',
        'algorithm.external_router.enabled':False,'algorithm.external_router.evaluation_enabled':False,
        'ray_init._temp_dir':str(out/'ray')}
    for k,v in changes.items():OmegaConf.update(cfg,k,v,force_add=True)
    OmegaConf.save(cfg,out/'seen_config.yaml')
    files={p.name:sha(p) for p in sorted(model.iterdir()) if p.is_file() and p.suffix in ('.json','.safetensors','.jinja','.txt')}
    receipt={'status':'passed','model':str(model),'model_files':files,'assets':str(assets),'step':a.step,
        'task_manifest_sha256':sha(out/'tasks_alfworld_seen140.json'),
        'source_model_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
    (out/'cpu_acceptance.json').write_text(json.dumps(receipt,indent=2))
    sys.path.insert(0,str(ROOT/'src/evaluation'))
    sys.path.insert(0,str(ROOT/'.deps/sdar'))
    import evaluate
    evaluate.HERE=out
    evaluate.main()
if __name__=='__main__':main()
