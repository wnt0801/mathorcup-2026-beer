"""One-time audited freezing of historical policies, not a retraining shortcut."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from datetime import datetime,timezone
from integrity import read_manifest,verify_manifest,normalize_manifest,sha

ROOT=Path(__file__).resolve().parents[1]

def main():
    old=ROOT/'outputs/competition_completion_v1'
    frozen=ROOT/'frozen_inputs'
    if frozen.exists():raise FileExistsError('Preserving existing frozen_inputs; preparation is one-time')
    legacy=read_manifest(old/'original_sha256.json')
    audited=verify_manifest(ROOT,legacy)
    target=ROOT/'tmp/review_v2_preservation_before.json';target.parent.mkdir(exist_ok=True)
    protect=[p for f in ['data','notebooks','outputs/competition_completion_v1'] for p in (ROOT/f).rglob('*') if p.is_file()]
    protect+=list((ROOT/'outputs').glob('*.zip'))
    target.write_text(json.dumps({p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in protect},ensure_ascii=False,indent=2),encoding='utf-8')
    frozen.mkdir()
    normalize_manifest(old/'original_sha256.json',frozen/'legacy_original_sha256.json')
    required={k:v for k,v in legacy.items() if k.startswith('data/')}
    provenance=[]
    for name in ['model_selection_cv_50.csv','test_predictions_50.csv']:
        src=ROOT/'outputs'/name;key='outputs/'+name
        shutil.copy2(src,frozen/name)
        required['frozen_inputs/'+name]=legacy[key]
        provenance.append({'input':name,'role':'frozen training-CV selection' if name.startswith('model_') else 'historical prediction comparison only; not training labels','source':key,'expected_sha256':legacy[key]})
    public=read_manifest(old/'support_file_manifest.json')
    key='outputs/competition_completion_v1/reference_supplement.xlsx'
    verify_manifest(ROOT,{key:public[key]})
    shutil.copy2(old/'reference_supplement.xlsx',frozen/'reference_supplement.xlsx')
    required['frozen_inputs/reference_supplement.xlsx']=public[key]
    provenance.append({'input':'reference_supplement.xlsx','role':'read-only public source audit; not model supervision','source':key,'expected_sha256':public[key]})
    (frozen/'source_sha256.json').write_text(json.dumps(required,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (frozen/'provenance.json').write_text(json.dumps({'prepared_utc':datetime.now(timezone.utc).isoformat(),'git_status_at_preparation':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'policy':'Expected data digests copied from v1 manifest; path normalization only. Selection and comparison digests retain historical values. No silent rebaselining.','legacy_audit':audited,'inputs':provenance},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Audited source freeze:',len(required),'required inputs;',len(audited),'legacy files verified')

if __name__=='__main__':main()
