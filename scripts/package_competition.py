"""Create local support archive without caches, review rasters or private environment files."""
from pathlib import Path
import hashlib
import json
import zipfile
from datetime import datetime,timezone
from uuid import uuid4
from review_config import ROOT,OUT

def main():
    import verify_competition
    verify_competition.main()
    files=[]
    for folder in ['data','frozen_inputs','scripts']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file():continue
            relative=p.relative_to(ROOT)
            if any(part.startswith('.') or part=='__pycache__' for part in relative.parts):continue
            if p.suffix=='.zip':continue
            files.append(p)
    for p in OUT.rglob('*'):
        if not p.is_file() or any(part.startswith('.') for part in p.relative_to(OUT).parts):continue
        if p.suffix=='.zip' or 'package_manifests' in p.parts or 'pdf_review' in p.parts:continue
        files.append(p)
    for name in ['README.md','README_competition.md','REPRODUCIBILITY.md','REVIEW_SUMMARY.md','requirements.txt','requirements_competition.txt']:
        p=ROOT/name
        if p.is_file():files.append(p)
    if (ROOT/'notebooks/STATUS.md').is_file():files.append(ROOT/'notebooks/STATUS.md')
    files=sorted(set(files))
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid4().hex[:6]
    archive=ROOT/'outputs'/('mathorcup_support_v2_'+tag+'.zip')
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('SUPPORT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path,digest in manifest.items():assert hashlib.sha256(z.read(path)).hexdigest()==digest
    record={'archive':archive.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files),'verified_crc':True,'verified_individual_sha256':True,'verified_utc':datetime.now(timezone.utc).isoformat(),'manifest':manifest}
    directory=OUT/'package_manifests';directory.mkdir(exist_ok=True)
    (directory/(archive.stem+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Support archive verified:',archive,'bytes=',archive.stat().st_size,'files=',len(files))
    return record

if __name__=='__main__':main()
