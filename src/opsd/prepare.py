"""Compose portable SDAR configuration and fixed seen140 inputs; CPU only."""
import argparse, hashlib, json, os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from ablation import contract, source_binding

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',type=Path,required=True,help='ALFWorld data root, containing json_2.1.1 and logic')
    p.add_argument('--model',type=Path,required=True,help='Route-SFT step2000 directory; base directory for baseline')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--cpus',type=int,default=22)
    a=p.parse_args()
    if a.baseline:p.error('This isolated repository runs its declared ablation only')
    import pandas as pd
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    assets,model,run=a.assets.resolve(),a.model.resolve(),a.run_dir.resolve()
    if run.exists():raise ValueError('Use a new run directory; never overwrite a training run')
    tasks=json.loads((ROOT/'configs/seen140_tasks.json').read_text())
    assert len(tasks)==140 and len({t['task_id'] for t in tasks})==140
    for i,t in enumerate(tasks):
        game=(assets/t['task_id']).resolve()
        assert game.is_relative_to(assets/'json_2.1.1/valid_seen') and t['seed']==2026+i
        assert sha(game)==t['game_sha256'],'Game asset version mismatch'
    assert (model/'config.json').is_file() and list(model.glob('*.safetensors'))
    recipe=json.loads((ROOT/'configs/sft.json').read_text(encoding='utf-8'))
    spec=contract()
    external=spec['variant']=='external_api_router'
    if a.baseline or external:
        receipt=json.loads((model/'parent_download_receipt.json').read_text(encoding='utf-8'))
        assert receipt['model_id']==recipe['base_model'] and receipt['revision']==recipe['base_revision']
        assert receipt['files'],'Empty parent receipt'
        for rec in receipt['files']:assert sha(model/rec['name'])==rec['sha256']
    else:
        m=json.loads((model/'sft_stage_manifest.json').read_text())
        assert m['stage']=='semantic_hr_route_bce_sft' and m['step']==2000
        assert m['checkpoint_complete'] and m['binding']['config']==recipe
        data_lock=json.loads((ROOT/'configs/data.json').read_text(encoding='utf-8'))
        assert m['binding']['dataset_manifest_sha256']==data_lock['files']['route_sft/manifest.json']
        for name,rec in m['model_files'].items():assert sha(model/name)==rec['sha256']
    os.environ['ALFWORLD_DATA']=str(assets)
    sys.path.insert(0,str(ROOT/'.deps/sdar'))
    import yaml
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
    c=yaml.safe_load((ROOT/'.deps/sdar/agent_system/environments/env_package/alfworld/configs/config_tw.yaml').read_text())
    env=get_environment(c['env']['type'])(c,train_eval='train')
    train_games={Path(x).resolve() for x in env.game_files}
    assert len(train_games)==3553,'Unexpected training task pool; inspect assets before training'
    assert not train_games & {(assets/t['task_id']).resolve() for t in tasks}
    run.mkdir(parents=True,exist_ok=False)
    for split,count in [('train',16),('val',140)]:
        items=[{'data_source':'text','prompt':[{'role':'user','content':''}],'ability':'agent',
                'extra_info':{'split':split,'index':i}} for i in range(count)]
        pd.DataFrame(items).to_parquet(run/f'{split}.parquet',index=False)
    with initialize_config_dir(config_dir=str(ROOT/'.deps/sdar/verl/trainer/config'),version_base=None):
        cfg=compose(config_name='rethink_public')
    changes={'data.train_files':str(run/'train.parquet'),'data.val_files':str(run/'val.parquet'),
        'actor_rollout_ref.model.path':str(model),'algorithm.local_hints.enabled':not a.baseline,
        'algorithm.local_hints.api_auth_file':str(run/'private/api_auth.json'),
        'algorithm.sdar.skills_dir':str(ROOT/'.deps/sdar/skills/alfworld'),
        'env.alfworld.fixed_eval_manifest':str(ROOT/'configs/seen140_tasks.json'),
        'trainer.default_local_dir':str(run/'checkpoints'),'trainer.rollout_data_dir':str(run/'rollouts'),
        'trainer.validation_data_dir':str(run/'validation'),'ray_init.num_cpus':a.cpus,
        'ray_init._temp_dir':str(run/'ray')}
    for k,v in changes.items():OmegaConf.update(cfg,k,v,force_add=True)
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg,run/'resolved_config.yaml')
    (run/'setup.json').write_text(json.dumps({'baseline':a.baseline,'assets':str(assets),'model':str(model),
        'recipe_sha256':sha(ROOT/'configs/opsd.yaml'), 'gpus':1, 'ablation':spec['variant'],
        'ablation_config_sha256':sha(ROOT/'configs/ablation.json'), 'source_binding':source_binding(),
        'training_games':len(train_games),'evaluation_games':140,'manifest_sha256':sha(ROOT/'configs/seen140_tasks.json')},indent=2))
    print('Prepared inputs and resolved configuration. No GPU/API/training started.')

if __name__=='__main__':main()
