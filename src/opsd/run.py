"""Explicit foreground launch. Credentials never enter Hydra config or command line."""
import argparse, json, os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--allow-api-cost',action='store_true');p.add_argument('--durable-every-update',action='store_true')
    a=p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('Set an explicitly allocated CUDA_VISIBLE_DEVICES')
    run=a.run_dir.resolve(); setup=json.loads((run/'setup.json').read_text())
    sys.path.insert(0,str(ROOT/'src'))
    from gpu_layout import devices
    devices(os.environ['CUDA_VISIBLE_DEVICES'],1)
    import hashlib
    if setup.get('recipe_sha256') != hashlib.sha256((ROOT/'configs/opsd.yaml').read_bytes()).hexdigest():
        raise RuntimeError('Saved run uses a different recipe; use a separate run directory')
    from ablation import contract, source_binding
    spec=contract()
    if setup.get('source_binding') != source_binding():
        raise RuntimeError('Source code changed after preparation; old checkpoints must not be silently reused')
    if setup.get('ablation') != spec['variant'] or setup.get('ablation_config_sha256') != hashlib.sha256((ROOT/'configs/ablation.json').read_bytes()).hexdigest():
        raise RuntimeError('A different ablation or stale setup cannot be resumed')
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(run/'resolved_config.yaml')
    if (cfg.trainer.n_gpus_per_node, cfg.trainer.nnodes, cfg.ray_init.num_gpus) != (1,1,1):
        raise RuntimeError('Expected one GPU in this isolated ablation')
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Assign exactly one usable GPU')
    cfg.ray_init.address='local'
    os.environ['RAY_ADDRESS']='local'
    if cfg.algorithm.local_hints.enabled:
        if not a.allow_api_cost:p.error('Method training uses a paid/external hint API; --allow-api-cost is required')
        token=os.environ.get('RETHINK_API_KEY')
        if not token or not os.environ.get('RETHINK_API_ENDPOINT'):p.error('Supply API key and HTTPS chat-completions endpoint through environment')
        from api_worker import validate_endpoint
        validate_endpoint(os.environ['RETHINK_API_ENDPOINT'])
        private=run/'private';private.mkdir(exist_ok=True,mode=0o700)
        path=private/'api_auth.json'
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as f:json.dump({'token':token},f)
        path.chmod(0o600)
        # The CPU worker reads this private file; Ray/Hydra must not serialize
        # the key as a runtime-environment value or expose it in config logs.
        os.environ.pop('RETHINK_API_KEY',None)
        os.environ['RETHINK_API_WORKER']=str(ROOT/'src/opsd/api_worker.py')
        if spec['variant']=='external_api_router':
            cfg.algorithm.external_router.api_auth_file=str(path)
            os.environ['RETHINK_ROUTER_WORKER']=str(ROOT/'src/opsd/router_worker.py')
    if a.durable_every_update:
        cfg.trainer.durable_recovery=True;cfg.trainer.save_freq=1
    sys.path.insert(0,str(ROOT/'src'))
    from runtime_paths import new_ray_directory
    cfg.ray_init._temp_dir=str(new_ray_directory())
    (run/'runtime_paths.json').write_text(json.dumps({'ray':cfg.ray_init._temp_dir}),encoding='utf-8')
    os.environ['ALFWORLD_DATA']=setup['assets']
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['PYTHONPATH']=str(ROOT/'.deps/sdar')
    sys.path.insert(0,str(ROOT/'.deps/sdar'))
    os.chdir(ROOT/'.deps/sdar')
    from verl.trainer.main_sdar import run_sdar
    import ray
    try:
        run_sdar(cfg)
    finally:
        ray.shutdown()

if __name__=='__main__':main()
