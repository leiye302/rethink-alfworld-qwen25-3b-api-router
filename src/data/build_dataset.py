"""Offline, deterministic selection. No API, model, GPU or semantic relabeling."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
from action_targets import assemble_target

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261013

def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def select(items, label, count, salt):
    pool=[x for x in items if x['route']==label]
    random.Random(f'{SEED}:{salt}:{label}').shuffle(pool)
    if len(pool)<count: raise ValueError(f'Insufficient {label}; never fabricate labels to fill quotas')
    return pool[:count]

def build(pool):
    candidates, excluded=[],[]
    for row in pool:
        label=row['final_label']
        reason='reference_review_quarantine' if row['reference_quarantine'] else None
        if label=='ignore': reason='semantic_ignore'
        if not row['include_entire_row'] or not row['eligible_for_training']:
            reason=reason or 'review_excluded'
        if reason:
            excluded.append({'audit_id':row['audit_id'],'reason':reason});continue
        if label not in ('good','bad'): raise ValueError('Unreviewed label')
        prefix=row['assistant_prefix']
        if not re.fullmatch(r'<think>\s*\S[\s\S]*</think>\s*',prefix):
            excluded.append({'audit_id':row['audit_id'],'reason':'invalid_reasoning_prefix'});continue
        if any(tag in prefix for tag in ('<RETHINK>','<action>','</action>')):
            excluded.append({'audit_id':row['audit_id'],'reason':'draft_action_or_marker_in_reasoning_prefix'});continue
        try: target=assemble_target(row,label)
        except ValueError:
            excluded.append({'audit_id':row['audit_id'],'reason':'action_evidence_failed'});continue
        fields=['audit_id','prompt','assistant_prefix','task_id','task_type',
                'source_executed_action','oracle_action','admissible_actions_before']
        candidates.append({**{k:row[k] for k in fields},'task_kind':'actor_route',**target})
    groups=defaultdict(list)
    for row in candidates: groups[(row['prompt'],row['assistant_prefix'])].append(row)
    candidates=[]
    for group in groups.values():
        if len({x['task_id'] for x in group})>1 or len({x['route'] for x in group})>1:
            excluded.extend({'audit_id':x['audit_id'],'reason':'conflicting_or_cross_task_prefix'} for x in group)
        else:
            ordered=sorted(group,key=lambda x:x['audit_id']);candidates.append(ordered[0])
            excluded.extend({'audit_id':x['audit_id'],'reason':'duplicate_visible_prefix'} for x in ordered[1:])
    by_task=defaultdict(list)
    for row in candidates: by_task[row['task_id']].append(row)
    types=sorted({x['task_type'] for x in candidates})
    if len(types)!=6: raise ValueError('Expected six ALFWorld families')
    for attempt in range(10000):
        rng=random.Random(f'{SEED}:dev-tasks:{attempt}')
        trial=[rng.choice(sorted(t for t,rs in by_task.items() if rs[0]['task_type']==kind)) for kind in types]
        held=[x for x in candidates if x['task_id'] in trial]
        rest=[x for x in candidates if x['task_id'] not in trial]
        hc,rc=Counter(x['route'] for x in held),Counter(x['route'] for x in rest)
        if hc['good']>=128 and hc['bad']>=128 and rc['good']>=10000 and rc['bad']>=6000: break
    else: raise ValueError('Not enough task-disjoint data')
    train=select(rest,'good',10000,'train')+select(rest,'bad',6000,'train')
    val=select(held,'good',128,'val')+select(held,'bad',128,'val')
    random.Random(SEED).shuffle(train);random.Random(SEED+1).shuffle(val)
    generation=select(val,'good',16,'generation')+select(val,'bad',16,'generation')
    for split,rs in [('train',train),('val',val)]:
        for row in rs: row['split']=split
    return train,val,generation,excluded,dict(Counter(x['route'] for x in candidates))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'data/route_sft')
    a=p.parse_args()
    pool_path=ROOT/'data/reviewed_pool.jsonl.gz'
    with gzip.open(pool_path,'rt',encoding='utf-8') as f: pool=[json.loads(x) for x in f]
    if len({x['audit_id'] for x in pool})!=len(pool): raise ValueError('Duplicate reviewed ID')
    train,val,gen,excluded,counts=build(pool)
    frozen=json.loads((ROOT/'data/selection_ids.json').read_text())
    actual={k:[x['audit_id'] for x in rs] for k,rs in [('train',train),('val',val),('generation',gen)]}
    assert actual==frozen,'Selection or order changed; do not silently continue'
    assert not {x['task_id'] for x in train}&{x['task_id'] for x in val}
    a.out.mkdir(parents=True,exist_ok=False)
    for name,rs in [('train',train),('val',val),('generation_val32',gen),('excluded',excluded)]:
        with (a.out/f'{name}.jsonl').open('w',encoding='utf-8',newline='\n') as f:
            for x in rs: f.write(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')
    manifest={'schema':'public_hr_route_dataset_v1','seed':SEED,'train_counts':dict(Counter(x['route'] for x in train)),
        'val_counts':dict(Counter(x['route'] for x in val)),'candidate_counts':counts,
        'excluded_counts':dict(Counter(x['reason'] for x in excluded)),
        'train_task_count':len({x['task_id'] for x in train}),'val_task_ids':sorted({x['task_id'] for x in val}),
        'task_overlap':0,'pool_sha256':digest(pool_path),'provenance':[],
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(a.out.glob('*.jsonl'))}}
    write_json(a.out/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','val_task_ids')}))

if __name__=='__main__':main()
