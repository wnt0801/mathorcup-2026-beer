"""Strict source hashes with an explicit, narrow Git LF/CRLF exception.

Never change a historical expected digest to make verification pass.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json

TEXT_SUFFIXES={'.csv','.txt','.md','.json','.ipynb','.graphml','.py','.yml','.yaml'}


def relative_key(value):
    key=str(value).replace('\\','/')
    p=PurePosixPath(key)
    if p.is_absolute() or '..' in p.parts or ':' in key or not p.parts:
        raise ValueError('Unsafe relative manifest path: '+key)
    return p.as_posix()


def read_manifest(path):
    raw=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    raw=raw.get('files',raw)
    result={}
    for k,v in raw.items():
        k=relative_key(k)
        if k in result:raise ValueError('Duplicate normalized manifest key: '+k)
        result[k]=v['sha256'] if isinstance(v,dict) else v
    return result


def sha(data):return hashlib.sha256(data).hexdigest()


def inspect_file(path,expected):
    path=Path(path);data=path.read_bytes();actual=sha(data)
    row={'expected_sha256':expected,'actual_sha256':actual,'status':'exact','canonical_lf_sha256':None}
    if path.suffix.lower() in TEXT_SUFFIXES:
        try:
            data.decode('utf-8-sig')
            if b'\x00' in data:raise UnicodeError('binary NUL')
        except UnicodeError:
            pass
        else:
            lf=data.replace(b'\r\n',b'\n')
            # Lone CR is not an allowed text normalization.
            if b'\r' not in lf:
                row['canonical_lf_sha256']=sha(lf)
                variants={'LF':sha(lf),'CRLF':sha(lf.replace(b'\n',b'\r\n'))}
                if actual!=expected and expected in variants.values():
                    row['status']='eol_only'
                    row['historical_form']=next(k for k,v in variants.items() if v==expected)
    if actual!=expected and row['status']!='eol_only':
        raise ValueError(f'CONTENT HASH MISMATCH: {path}; expected={expected}; actual={actual}')
    return row


def verify_manifest(root,manifest):
    values=read_manifest(manifest) if isinstance(manifest,(str,Path)) else {relative_key(k):v for k,v in manifest.items()}
    rows=[]
    for key,expected in values.items():
        p=Path(root)/key
        if not p.is_file():raise FileNotFoundError('Missing protected input: '+key)
        rows.append({'path':key,**inspect_file(p,expected)})
    return rows


def normalize_manifest(source,target):
    """Path-only migration: all expected digests remain unchanged."""
    values=read_manifest(source)
    Path(target).write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
