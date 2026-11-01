"""Portable OpenAI-compatible transport; reuses the method's exact hint protocol.

No GPU, quality gate, environment mutation, automatic provider switch or secrets
in outputs. Malformed content is recorded and falls back via the native parser;
transport/auth failures hold the batch rather than silently dropping hints.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib, http.client, importlib.util, json, os, time, uuid
from pathlib import Path
import urllib.error, urllib.parse, urllib.request

ROOT=Path(__file__).resolve().parents[2]
protocol=None
AUTH_HEADERS={'User-Agent':'rethink-external-router/2.0', 'x-opencode-session':'rethink-'+uuid.uuid4().hex}


def read_auth(path):
    path=Path(path)
    if os.name!='nt' and path.stat().st_mode&0o077:
        raise RuntimeError('AUTH_PERMISSIONS_TOO_OPEN')
    value=json.loads(path.read_text(encoding='utf-8'))
    token=value.get('token')
    if not isinstance(token,str) or not token.strip() or any(c in token for c in '\r\n'):
        raise RuntimeError('INVALID_API_AUTH')
    for field,header in [('session','x-opencode-session'),('org_id','x-opencode-org-id')]:
        item=value.get(field)
        if item is not None:
            if not isinstance(item,str) or not item.strip() or len(item)>200 or any(c in item for c in '\r\n'):
                raise RuntimeError('INVALID_API_HEADER')
            AUTH_HEADERS[header]=item
    return token

def load_protocol():
    global protocol
    if protocol is None:
        path=ROOT/'.deps/sdar/verl/trainer/ppo/rethink_api_protocol.py'
        if not path.is_file():
            raise RuntimeError('PINNED_FRAMEWORK_REQUIRED_RUN_SETUP')
        spec=importlib.util.spec_from_file_location('public_hint_protocol',path)
        protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
    return protocol

def validate_endpoint(endpoint):
    parsed=urllib.parse.urlsplit(endpoint)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError('HTTPS_ENDPOINT_WITHOUT_EMBEDDED_SECRETS_REQUIRED')
    return endpoint

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    temp.replace(path)

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise RuntimeError('API_REDIRECT_REFUSED')

def request_json(body,token,endpoint):
    """HTTPS transport with honest identification and stable private session."""
    handlers=[NoRedirect]
    if os.environ.get('RETHINK_API_TRUST_ENV','true').lower() in {'0','false','no'}:
        handlers.append(urllib.request.ProxyHandler({}))
    opener=urllib.request.build_opener(*handlers)
    data=json.dumps(body,ensure_ascii=False).encode()
    for attempt in range(6):
        try:
            request=urllib.request.Request(endpoint,data=data,headers={
                **AUTH_HEADERS,'Authorization':'Bearer '+token,'Content-Type':'application/json'},method='POST')
            with opener.open(request,timeout=180) as response:result=json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (408,429,500,502,503,504):raise RuntimeError('API_AUTH_OR_REQUEST_HOLD') from None
        except (urllib.error.URLError,TimeoutError,ConnectionError,http.client.RemoteDisconnected,json.JSONDecodeError):pass
        if attempt==5:raise RuntimeError('API_TRANSIENT_HOLD')
        time.sleep(min(60,2**(attempt+1)))
    if not isinstance(result,dict) or not isinstance(result.get('choices'),list) or not result['choices']:
        raise RuntimeError('API_MALFORMED_RESPONSE_HOLD')
    return result

def execute(group,cache,token,endpoint):
    load_protocol()
    body=protocol.request_body(group)
    if group.get('route_source')=='external_api':
        # Only selection provenance changes; the hint instructions, model,
        # trajectory grouping and admissibility checks remain the mother method.
        system=body['messages'][0]['content']
        original='The Actor has ALREADY emitted RETHINK at every requested target step.'
        if system.count(original)!=1:raise RuntimeError('HINT_ROUTE_PROMPT_CONTRACT_CHANGED')
        body['messages'][0]['content']=system.replace(original,
            'An external reasoning-only judge has selected every requested target step.')
    identity=protocol.digest({'body':body,'protocol':protocol.VERSION,'endpoint':endpoint})
    path=cache/(identity+'.json')
    if path.exists():
        value=json.loads(path.read_text());assert value['request_sha256']==identity
        value['outputs']=protocol.parse_hints(value.get('content'),group,value.get('finish_reason'))
        return value,False
    result=request_json(body,token,endpoint)
    choice=result['choices'][0]
    if not isinstance(choice,dict) or not isinstance(choice.get('message'),dict):
        raise RuntimeError('API_MALFORMED_RESPONSE_HOLD')
    content=choice.get('message',{}).get('content');finish=choice.get('finish_reason')
    value={'request_sha256':identity,'schema':protocol.VERSION,'response_model':result.get('model'),
        'content':content,'finish_reason':finish,'usage':result.get('usage',{}),
        'outputs':protocol.parse_hints(content,group,finish)}
    save(path,value)
    return value,True

def main():
    p=argparse.ArgumentParser();p.add_argument('request',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--auth',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--max-usd',type=float,default=0,help='Compatibility argument, not billing control')
    a=p.parse_args()
    endpoint=validate_endpoint(os.environ.get('RETHINK_API_ENDPOINT',''))
    load_protocol()
    token=read_auth(a.auth)
    payload=json.loads(a.request.read_text());groups=protocol.group_requests(payload['requests'])
    if payload.get('route_source')=='external_api':
        for group in groups:group['route_source']='external_api'
    a.cache.mkdir(parents=True,exist_ok=True)
    workers=int(os.environ.get('RETHINK_API_CONCURRENCY','4'))
    if not 1<=workers<=4:raise ValueError('Use 1 to 4 API workers')
    outputs={};usage=[];calls=0;start=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(execute,g,a.cache,token,endpoint) for g in groups]
        for future in as_completed(futures):
            result,called=future.result();outputs.update(result['outputs']);usage.append(result['usage']);calls+=called
    save(a.output,{'request_sha256':payload['request_sha256'],'outputs':outputs,
        'grouped_requests':len(groups),'api_calls_this_run':calls,'schema':protocol.VERSION,
        'prompt_tokens':sum(u.get('prompt_tokens',0) for u in usage),
        'completion_tokens':sum(u.get('completion_tokens',0) for u in usage),
        'elapsed_s':time.monotonic()-start,'actor_route_changed_by_analyzer':False,'student_sees_hint':False,
        'billing_cost_tracked':False})

if __name__=='__main__':
    try:main()
    except RuntimeError as exc:
        # Allowlisted codes only; never output provider responses or auth data.
        code=str(exc)
        print(code if code.isupper() and code.replace('_','').isalnum() else 'API_WORKER_HOLD')
        raise SystemExit(75 if code=='API_TRANSIENT_HOLD' else 76)
    except Exception:
        print('API_WORKER_INVALID_INPUT_OR_CACHE')
        raise SystemExit(76)
