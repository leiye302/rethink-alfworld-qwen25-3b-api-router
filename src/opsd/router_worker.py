"""One H/R per API request, with exact cache binding and no test-time route."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import time

import api_worker
from router_protocol import VERSION,digest,parse_result,request_body


def execute(item,cache,token,endpoint,model):
    body=request_body(item,model)
    key=digest({'body':body,'endpoint':endpoint,'protocol':VERSION})
    path=cache/(key+'.json')
    if path.exists():
        value=json.loads(path.read_text(encoding='utf-8'))
        if value['request_sha256']!=key:raise RuntimeError('ROUTER_CACHE_INTEGRITY_FAILURE')
        return {**parse_result(value['content'],item['R'],value['finish_reason']),
                'usage':{},'cache_hit':True,'response_model':value.get('response_model')}
    result=api_worker.request_json(body,token,endpoint)
    try:
        choice=result['choices'][0]
        content=choice['message']['content'];finish=choice.get('finish_reason')
    except (KeyError,IndexError,TypeError):
        raise RuntimeError('API_MALFORMED_RESPONSE_HOLD') from None
    usage=result.get('usage',{})
    value={'request_sha256':key,'content':content,'finish_reason':finish,
           'response_model':result.get('model'),'usage':usage if isinstance(usage,dict) else {}}
    api_worker.save(path,value)
    return {**parse_result(content,item['R'],finish),'usage':value['usage'],'cache_hit':False,
            'response_model':value['response_model']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--auth',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    endpoint=api_worker.validate_endpoint(os.environ.get('RETHINK_API_ENDPOINT',''))
    if os.name!='nt' and args.auth.stat().st_mode&0o077:raise RuntimeError('AUTH_PERMISSIONS_TOO_OPEN')
    token=json.loads(args.auth.read_text())['token']
    payload=json.loads(args.request.read_text())
    if payload['schema']!=VERSION:raise RuntimeError('ROUTER_PROTOCOL_MISMATCH')
    if digest({k:v for k,v in payload.items() if k!='request_sha256'})!=payload['request_sha256']:
        raise RuntimeError('ROUTER_REQUEST_INTEGRITY_FAILURE')
    keys=[item['key'] for item in payload['requests']]
    if len(keys)!=len(set(keys)) or any(item['key']!=digest({'H':item['H'],'R':item['R']}) for item in payload['requests']):
        raise RuntimeError('ROUTER_DUPLICATE_OR_UNBOUND_ROW')
    workers=int(payload['concurrency'])
    if not 1<=workers<=4:raise ValueError('Use 1 to 4 router workers')
    args.cache.mkdir(parents=True,exist_ok=True)
    outputs={};started=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(execute,item,args.cache,token,endpoint,payload['model']):item['key']
                 for item in payload['requests']}
        for future in as_completed(futures):outputs[futures[future]]=future.result()
    api_worker.save(args.output,{'request_sha256':payload['request_sha256'],'schema':VERSION,
        'outputs':outputs,'api_calls_this_run':sum(not v['cache_hit'] for v in outputs.values()),
        'prompt_tokens':sum(v['usage'].get('prompt_tokens',0) for v in outputs.values()),
        'completion_tokens':sum(v['usage'].get('completion_tokens',0) for v in outputs.values()),
        'seconds':time.monotonic()-started,'judge_saw_current_action':False,
        'judge_saw_future_or_episode':False,'student_modified':False})


if __name__=='__main__':
    try:main()
    except RuntimeError as exc:
        code=str(exc)
        print(code if code.isupper() and code.replace('_','').isalnum() else 'ROUTER_WORKER_FAILURE')
        raise SystemExit(75 if code=='API_TRANSIENT_HOLD' else
                         76 if code in {'API_AUTH_OR_REQUEST_HOLD','API_MALFORMED_RESPONSE_HOLD'} else 77)
    except Exception:
        print('ROUTER_WORKER_INVALID_INPUT_OR_CACHE')
        raise SystemExit(77)
