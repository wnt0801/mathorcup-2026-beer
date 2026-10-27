"""One portable candidate package; only current, required inputs/code/results."""
from pathlib import Path,PurePosixPath
import hashlib,json,posixpath,re,zipfile,subprocess,sys,tempfile,time,os
from datetime import datetime,timezone
from review_config import ROOT,OUT

ACTIVE_SCRIPTS=['run_review.py','competition_pipeline.py','review_audits.py','review_report.py','build_competition_deliverables.py','verify_competition.py','package_competition.py','review_config.py','integrity.py','test_review_contracts.py']
JSON_RESULTS=['analysis_design.json','original_sha256.json','verification.json','dependencies_actual.json','run_summary.json','fresh_environment.json','q1_score_scale_evidence.json','q1_workbook_metadata.json','q1_scorecard_template_differences.json','q3_stale_guard_assessment.json','reference_provenance.json','references_verification.json','pdf_visual_review.json']
JSON_RESULTS += ['final_validation_execution.json','post_visual_build.json','font_evidence.json']


def portable_json(data):
    if isinstance(data,dict):
        result={}
        for k,v in data.items():
            if k=='verification_root':result[k]='.'
            elif k in ['output','output_relative']:result[k]='results'
            elif k in ['selected_font','python_executable']:result[k]=PurePosixPath(str(v).replace('\\','/')).name
            else:result[k]=portable_json(v)
        return result
    if isinstance(data,list):return [portable_json(x) for x in data]
    if isinstance(data,str):return data.replace(str(ROOT),'.').replace(str(ROOT).replace('\\','/'),'.')
    return data


def check_links(payload):
    checked=0
    for name,data in payload.items():
        if not name.endswith('.md'):continue
        text=data.decode('utf-8-sig')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            target=target.strip('<>').split('#',1)[0]
            if not target or target.startswith(('http:','https:','mailto:')):continue
            resolved=posixpath.normpath(posixpath.join(posixpath.dirname(name),target))
            if resolved not in payload:raise ValueError(f'Broken package link: {name} -> {target}')
            checked+=1
    return checked


def main():
    from verify_competition import main as verify
    verify()
    payload={}
    def put(path,name=None):payload[name or path.relative_to(ROOT).as_posix()]=path.read_bytes()
    for path in (ROOT/'data').iterdir():
        if path.is_file():put(path)
    for name in ['source_sha256.json','model_selection_cv_50.csv','test_predictions_50.csv','reference_supplement.xlsx']:
        put(ROOT/'frozen_inputs'/name)
    for name in ACTIVE_SCRIPTS:put(ROOT/'scripts'/name)
    put(ROOT/'requirements_competition.txt')
    for path in OUT.glob('*.csv'):put(path,'results/'+path.name)
    for path in OUT.glob('*.graphml'):put(path,'results/'+path.name)
    for name in JSON_RESULTS:
        p=OUT/name
        if p.is_file():payload['results/'+name]=json.dumps(portable_json(json.loads(p.read_text(encoding='utf-8'))),ensure_ascii=False,indent=2).encode('utf-8')
    for name in ['competition_paper.md','competition_paper.pdf','Q1_SCORING_SCOPE.md','TASK_ALIGNMENT.md']:
        put(OUT/name,'results/'+name)
    for p in (OUT/'figures').iterdir():
        if p.is_file() and p.suffix in ['.png','.svg','.json']:put(p,'results/figures/'+p.name)
    for p in OUT.glob('*.joblib'):put(p,'results/'+p.name)
    for p in (OUT/'review_notebooks').glob('*.ipynb'):put(p,'results/review_notebooks/'+p.name)
    payload['README.md']='''# MathorCup 候选提交材料 v3

推荐论文：[results/competition_paper.pdf](results/competition_paper.pdf)。
结果入口：[results/START_HERE.md](results/START_HERE.md)。
复现说明：[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
评分口径：[results/Q1_SCORING_SCOPE.md](results/Q1_SCORING_SCOPE.md)。

原始0–6强度未还原，Stale hops仅为弱模型代理，五项汇总公式未验证。当前归档就是唯一推荐支撑包，未包含任何历史版本、缓存或原机器绝对路径日志。
'''.encode('utf-8')
    payload['results/START_HERE.md']='''# 当前材料入口

[推荐论文](competition_paper.pdf) · [可编辑论文](competition_paper.md)

[三问对应关系](TASK_ALIGNMENT.md) · [评分口径](Q1_SCORING_SCOPE.md)

[运行命令](../REPRODUCIBILITY.md) · [审核总结](../REVIEW_SUMMARY.md)

结果复查Notebook在review_notebooks/；完整训练入口为scripts/run_review.py。
'''.encode('utf-8')
    payload['REPRODUCIBILITY.md']='''# 复现

从归档根目录运行，Python3.13；先创建新虚拟环境并安装requirements_competition.txt。

Windows PowerShell：
```powershell
py -3.13 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements_competition.txt
& ./.venv/Scripts/python.exe scripts/run_review.py
```

macOS/Linux（仅提供命令，不代表在这些系统实跑）：
```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements_competition.txt
.venv/bin/python scripts/run_review.py --font /path/to/ChineseTrueType.ttf
```

默认每次创建新的outputs/reproduction_时间戳目录，构建、验证、打包自动使用同一目录。已提供的results目录不会被覆盖。独立核验当前结果：python scripts/verify_competition.py --output results。输出含已冻结训练CV选择及历史预测比较文件，因此不声称只由附件重新选择全部模型。

需要中文TrueType字体。可用--font或MATHORCUP_CJK_FONT指定；没有可用中文字形时明确报错。系统字体不随包分发。

模型按200/50固定划分训练，种子42；参数不按已查看测试集继续选择。复查Notebook仅读结果，不替代完整脚本。归档内SUPPORT_MANIFEST.json记录每个文件的SHA256，JSON运行位置已作明确的可移植化处理，数值结果未改写。
'''.encode('utf-8')
    summary=ROOT/'REVIEW_SUMMARY.md'
    text=summary.read_text(encoding='utf-8') if summary.is_file() else '本轮代码和结果见results；完整历史AI交互导出未提供，不能补造。'
    text=text.replace('outputs/competition_candidate_v3/','results/')
    payload['REVIEW_SUMMARY.md']=text.encode('utf-8')
    links=check_links(payload)
    manifest={k:hashlib.sha256(v).hexdigest() for k,v in sorted(payload.items())}
    final=OUT/'submission_candidate_v3.zip'
    if final.exists():raise FileExistsError('Preserving existing candidate archive; rebuild only after explicitly archiving/replacing your own candidate.')
    with zipfile.ZipFile(final,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for k,v in payload.items():archive.writestr(k,v)
        archive.writestr('SUPPORT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(final) as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read(k)).hexdigest()==v for k,v in manifest.items())
        # Verify the actual extracted package, including dependencies such as GraphML.
        scratch=ROOT/'tmp';scratch.mkdir(exist_ok=True)
        extracted=Path(tempfile.mkdtemp(prefix='v3_package_',dir=scratch))
        assert all(not PurePosixPath(k).is_absolute() and '..' not in PurePosixPath(k).parts for k in archive.namelist())
        archive.extractall(extracted)
    assert all(hashlib.sha256((extracted/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
    begin=time.monotonic();started=datetime.now(timezone.utc).isoformat()
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    result=subprocess.run([sys.executable,str(extracted/'scripts/verify_competition.py'),'--output',str(extracted/'results')],cwd=extracted,env=env,capture_output=True,text=True,encoding='utf-8')
    (OUT/'.review/package_extracted_verify.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    extraction={'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'returncode':result.returncode,'seconds':time.monotonic()-begin,'file_hashes':'passed','log':'.review/package_extracted_verify.log'}
    (OUT/'.review/package_extracted_execution.json').write_text(json.dumps(extraction,indent=2),encoding='utf-8')
    if result.returncode:raise RuntimeError('Extracted package verification failed; see .review/package_extracted_verify.log')
    record={'archive':final.name,'sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'files':len(payload),'checked_markdown_links':links,'verified_utc':datetime.now(timezone.utc).isoformat(),'CRC':'passed','file_hashes':'passed','policy':'current required inputs, source and results only; no history/cache/screenshots/absolute-path logs','manifest':manifest}
    record['extracted_verification']=extraction
    (OUT/'package_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Candidate package verified:',final,'files=',len(payload),'links=',links)
    return record

if __name__=='__main__':main()
