"""One recommended command: recompute, audit, build review materials, verify.

    python scripts/run_review.py --output outputs/a_new_run --font FONT.ttf
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--font');parser.add_argument('--no-package',action='store_true');args=parser.parse_args()
    destination=args.output or os.environ.get('MATHORCUP_OUTPUT') or 'outputs/reproduction_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen_training_models.joblib').exists():
        raise FileExistsError('Preserving completed run. Choose a new --output directory.')
    os.environ['MATHORCUP_OUTPUT']=str(out)
    if args.font:os.environ['MATHORCUP_CJK_FONT']=str(Path(args.font).resolve())
    from review_config import chinese_font
    chinese_font()
    records=[];started=datetime.now(timezone.utc).isoformat()
    (out/'.review').mkdir(exist_ok=True)
    for name,command in [
        ('dependency_check',[sys.executable,'-m','pip','check']),
        ('targeted_contract_tests',[sys.executable,str(ROOT/'scripts/test_review_contracts.py')]),
        ('pipeline',[sys.executable,str(ROOT/'scripts/competition_pipeline.py'),'--stage','all']),
        ('build',[sys.executable,str(ROOT/'scripts/build_competition_deliverables.py')]),
        ('verify',[sys.executable,str(ROOT/'scripts/verify_competition.py')]),
    ]:
        begin=time.monotonic();print('START',name,flush=True)
        log=out/'.review'/(name+'.log')
        with log.open('w',encoding='utf-8') as stream:
            env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['MPLCONFIGDIR']=str(out/'.matplotlib')
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
            code=process.wait()
        records.append({'stage':name,'command':command,'returncode':code,'seconds':time.monotonic()-begin,'log':log.name})
        (out/'.review/run_execution.json').write_text(json.dumps({'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'root':str(ROOT),'output':str(out),'python':sys.executable,'stages':records,'status':'passed' if code==0 and name=='verify' else 'running' if code==0 else 'failed'},ensure_ascii=False,indent=2),encoding='utf-8')
        if code:raise RuntimeError(f'{name} failed; inspect {log}')
    packages=['numpy','pandas','scipy','scikit-learn','openpyxl','joblib','networkx','jupyterlab','jupyter-client','nbformat','nbclient','ipykernel','matplotlib','reportlab','pypdf','pillow']
    (out/'dependencies_actual.json').write_text(json.dumps({'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in packages}},indent=2),encoding='utf-8')
    (out/'run_summary.json').write_text(json.dumps({'status':'passed','started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'output_relative':os.path.relpath(out,ROOT).replace('\\','/'),'stages':[{'stage':r['stage'],'returncode':r['returncode'],'seconds':r['seconds']} for r in records],'selection':'frozen policies and parameters unchanged'},ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.no_package:
        from package_competition import main as package
        package()
    print('Completed review workflow:',out,flush=True)

if __name__=='__main__':main()
