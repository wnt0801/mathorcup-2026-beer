"""Run the entire workflow in a new inputs-only copy; keep all copies for audit."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--font');args=parser.parse_args()
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid4().hex[:6]
    clean=ROOT/'tmp'/('review_clean_'+tag);clean.mkdir(parents=True,exist_ok=False)
    for folder in ['data','frozen_inputs','scripts']:
        shutil.copytree(ROOT/folder,clean/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['requirements.txt','requirements_competition.txt','README.md','README_competition.md','REPRODUCIBILITY.md']:
        shutil.copy2(ROOT/name,clean/name)
    assert not (clean/'outputs').exists()
    snapshot={p.relative_to(clean).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in clean.rglob('*') if p.is_file()}
    out=clean/'outputs/competition_completion_v2'
    command=[sys.executable,str(clean/'scripts/run_review.py'),'--output',str(out)]
    if args.font:command+=['--font',str(Path(args.font).resolve())]
    print('Clean inputs-only copy:',clean,flush=True)
    env=os.environ.copy();env.pop('MATHORCUP_OUTPUT',None);env['PYTHONIOENCODING']='utf-8'
    code=subprocess.call(command,cwd=clean,env=env)
    packaging=[]
    if code==0:
        env['MATHORCUP_OUTPUT']=str(out)
        for i in range(2):
            packcode=subprocess.call([sys.executable,str(clean/'scripts/package_competition.py')],cwd=clean,env=env)
            packaging.append({'attempt':i+1,'returncode':packcode})
            if packcode:
                code=packcode
                break
        archives=sorted((clean/'outputs').glob('mathorcup_support_v2_*.zip'))
        if code==0 and len(archives)!=2:raise AssertionError('Repeated package creation did not produce two distinct archives')
    else:
        archives=[]
    result={'verified_utc':datetime.now(timezone.utc).isoformat(),'clean_root':str(clean),'output':str(out),'returncode':code,'status':'passed' if code==0 else 'failed','inputs_only_initial_copy':True,'copied_legacy_outputs':False,'copied_pretrained_models':False,'fresh_environment_install':False,'environment_note':'Fresh project copy, using the current existing Python environment; no Linux/macOS runtime or empty-venv install claim.','packaging_runs':packaging,'new_archives':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in archives],'initial_file_sha256':snapshot}
    p=ROOT/'tmp'/('clean_reproduction_'+tag+'.json');p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if code==0:
        (out/'clean_reproduction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Clean reproduction record:',p,flush=True)
    if code:raise SystemExit(code)

if __name__=='__main__':main()
