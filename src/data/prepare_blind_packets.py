"""Export H/R-only review requests; never leak targets or outcomes to the judge."""
import argparse, gzip, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def packet(row):
    prefix=row['assistant_prefix']
    assert prefix.startswith('<think>') and prefix.rstrip().endswith('</think>')
    # Preserve interior whitespace; remove only exact outer boundary tags.
    import re
    r=re.sub(r'^<think>','',prefix,count=1)
    r=re.sub(r'</think>(\s*)$',r'\1',r,count=1)
    return {'audit_id':row['audit_id'],'H':row['prompt'],'R':r}
def messages(records):
    instruction=(ROOT/'src/data/prompts/hr_review.txt').read_text(encoding='utf-8')
    return [{'role':'system','content':instruction},{'role':'user','content':json.dumps(
        {'records':[{'id':i,'H':r['H'],'R':r['R']} for i,r in enumerate(records)]},ensure_ascii=False)}]
def validate_result(text, records):
    # An API response is a candidate, not permission to import it into the dataset.
    body=json.loads(text);labels=body['labels']
    assert len(labels)==len(records) and sorted(x['id'] for x in labels)==list(range(len(records)))
    out=[]
    for x in labels:
        assert set(x)=={'id','label','reason'} and x['label'] in ('good','bad','ignore')
        assert isinstance(x['reason'],str) and x['reason'].strip()
        out.append({'audit_id':records[x['id']]['audit_id'],'label':x['label'],'reason':x['reason']})
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with gzip.open(ROOT/'data/reviewed_pool.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(x) for x in f]
    with a.out.open('x',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(packet(r),ensure_ascii=False)+'\n')
    print('Blind packets written; no API call or label changes.')
if __name__=='__main__':main()
