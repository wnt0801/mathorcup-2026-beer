"""Create local support archive without caches, review rasters or private environment files."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/competition_completion_v1'

def main():
    import verify_competition
    verify_competition.main()
    files=[]
    for folder in ['data','notebooks','scripts','outputs']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file():continue
            relative=p.relative_to(ROOT)
            if any(part.startswith('.') or part=='__pycache__' for part in relative.parts):continue
            if p.suffix=='.zip':continue
            if p.name.startswith(('task_page-','paper_review-','review_contact_','final_review-','final_contact_','last_check-')):continue
            files.append(p)
    for name in ['README.md','README_competition.md','competition_plan.md','requirements.txt','requirements_competition.txt','build_defect_audit.py']:
        files.append(ROOT/name)
    files=sorted(set(files))
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='support_file_manifest.json'}
    mp=OUT/'support_file_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if mp not in files:files.append(mp)
    archive=ROOT/'outputs/mathorcup_support_v1.zip'
    if archive.exists():raise FileExistsError('Preserving existing archive: '+str(archive))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path,digest in manifest.items():assert hashlib.sha256(z.read(path)).hexdigest()==digest
    print('Support archive verified:',archive,'bytes=',archive.stat().st_size,'files=',len(files))

if __name__=='__main__':main()
