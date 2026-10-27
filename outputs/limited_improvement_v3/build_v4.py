"""Adopt only passing calibration dimensions; reuse unchanged v3 Q2/Q3 outputs."""
from pathlib import Path
import os,sys,json,shutil,importlib.util,hashlib,zipfile,subprocess,time
from datetime import datetime,timezone
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'scripts/competition_pipeline.py').is_file())
BASE=Path(os.environ.get('MATHORCUP_BASE',str(ROOT/'results' if (ROOT/'results').is_dir() else ROOT/'outputs/competition_candidate_v3'))).resolve()
OUT=HERE/'adopted_v4'

def build():
    assert not OUT.exists(),'Preserve an existing adopted version; choose a new experiment copy to rebuild.'
    decision=pd.read_csv(HERE/'q1_calibration_decision.csv');accepted=decision.loc[decision.adopt,'target'].tolist()
    assert accepted and not pd.read_csv(HERE/'q3_decision.csv').adopt.any()
    OUT.mkdir()
    for p in BASE.iterdir():
        if p.is_file() and p.suffix in ['.csv','.json','.joblib','.graphml']:
            name='base_'+p.name if p.name in ['verification.json','run_summary.json','q1_score_scale_evidence.json','pdf_visual_review.json','package_verification.json'] else p.name
            shutil.copy2(p,OUT/name)
    shutil.copytree(BASE/'figures',OUT/'figures');shutil.copytree(BASE/'review_notebooks',OUT/'review_notebooks')
    old=pd.read_csv(BASE/'q1_out_of_sample_predictions.csv');params=pd.read_csv(HERE/'q1_calibration_parameters.csv').set_index('target')
    cv=pd.read_csv(HERE/'q1_calibrated_training_oof.csv').set_index(['beer','target'])
    output=old.loc[old.target.isin(accepted)].copy()
    output['predicted_raw_panel_mean']=np.clip(output.target.map(params.intercept)+output.predicted_standardized*output.target.map(params.slope),0,6)
    for i,r in output.loc[output['split'].eq('training_oof')].iterrows():
        output.at[i,'predicted_raw_panel_mean']=cv.at[(r.beer,r.target),'predicted_raw_panel_mean']
    output['nearest_integer_grade']=np.floor(output.predicted_raw_panel_mean+.5).astype(int)
    output['grade_definition']='nearest integer of calibrated panel mean; provisional reporting rule, not official panel vote'
    output['calibration_scope']='fold-trained for OOF; full original200 for existing heldout predictions; no new test performance used'
    output.to_csv(OUT/'q1_calibrated_predictions_partial.csv',index=False,encoding='utf-8-sig')
    os.environ['MATHORCUP_OUTPUT']=str(OUT);sys.path.insert(0,str(ROOT/'scripts'))
    # Reuse the existing layout implementation with only the version label changed.
    layout=(ROOT/'scripts/build_competition_deliverables.py').read_text(encoding='utf-8').replace('候选提交版 v3','候选提交版 v4')
    (HERE/'v4_layout.py').write_text(layout,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('v4_layout',HERE/'v4_layout.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    import review_report
    title,sections=review_report.paper(m.table)
    title='啤酒感官评分校准、稳定关联与局部风格谱生成'
    replacements={
      '原始评分逆变换缺少依据，另提供明确标注的相对七级。':'新公开原始评分提供38维可复核配对，其中25维（Aroma/Flavour共15维）的训练内校准达到预定标准；其余维度保留标准化预测及相对七级。校准估计面板平均强度，不是逐人评分精确逆变换。',
      '原始评分卡除Beer#外相同，汇总为空白模板。附件4、7及公开补充表[3]没有提供可验证的完整原始评分与逆变换参数。即使假设z=(r-μ)/σ，未知μ、σ也不能唯一恢复原始r；相对分级、标准化预测和原始0–6强度是三个不同对象。':'赛题评分卡仍为空白模板。新公开的匿名原始评分[4]含2197条记录、16名评分者、250款、50维0–6评分。匿名beer_name不能直接连接附件beer；本轮用3维精确标准化数值指纹得到唯一250对映射，再以其他维度交叉核实。逐评分者标准化后聚合通常没有统一精确逆函数，本文仅采用验证支持的部分面板平均强度校准。',
      '原始强度评分需要主办方提供校准信息；本文没有对标准化值进行min-max或平移后冒称还原。':'本节仍为全部50维标准化预测和相对等级基线。新增原始强度校准结果独立列于下一节，不将相对等级冒充原始强度。',
      '第一问完成标准化连续预测和明确定义的相对七级，原始0–6强度仍需校准。':'第一问保留全50维标准化预测和相对七级，并新增25维有配对原始评分支持的面板平均强度校准，其中Aroma/Flavour为15维。其余维度不宣称已校准，官方整数分档及匿名ID映射仍需确认。',
      '第一问是否接受相对等级及目标维度清单；原始评分标准化/聚合参数；':'第一问接受原始面板均值、其整数分档还是相对等级，以及目标维度清单；匿名原始数据ID对应与12维未复现项目的预处理；',
      '本文不补造。代码和材料说明保留真实AI参与':'本文不补造。代码和材料说明保留真实AI参与',
    }
    cleaned=[]
    for h,b in sections:
        for a,c in replacements.items():b=b.replace(a,c)
        cleaned.append((h,b))
    sections=cleaned
    af=decision.loc[decision.adopt&decision.target.str.startswith(('A_','F_'))]
    selected=decision.loc[decision.adopt]
    text=f'''新原始评分的发布MD5与下载文件一致[4]。按评分者与维度用样本标准差标准化，再按啤酒平均；A_malt_all、F_malt_all、overall三个固定维度的最大配对误差小于5×10^-10，匹配均唯一且形成一一映射。原200款训练中38维可复现附件4，最大误差1.64×10^-8；其余12维不复现，不能按相关性接近强行纳入。原始记录和逐维差异均随包保留。

校准对象r为原始逐评分记录的平均强度。每维拟合r=a+bz，输出限制在评分卡0–6范围；a、b只用训练折配对的观测z及r拟合。化学预测用原冻结模型的折外预测，不改变模型选择。每折160款拟合、40款验证，沿用原200款内5折；原50款未用于此次选择。不是假定原始均值为3，也不是按min/max缩放。面板人员及场次不同意味着该映射是统计校准，不能恢复每个评分者的评分。

事先要求观测z的校准MAE不超过0.30分、化学预测经校准的MAE比原始训练均值下降至少10%、且至少4/5折改善。38维中25维达到标准，其中Aroma/Flavour共15维。通过25维的平均MAE为{selected.prediction_MAE.mean():.4f}分，原始均值基准{selected.baseline_MAE.mean():.4f}；Aroma/Flavour通过维度为{af.prediction_MAE.mean():.4f}对{af.baseline_MAE.mean():.4f}。这些是用于采纳的训练内成绩，存在筛选乐观性，不能作为新的独立测试结论。

下面保留全部38个可重建维度，未通过者也展示。另12维未复现：A_hops_woody、A_esters_flower、A_esters_fruity、F_hops_tropical、F_hops_woody、F_esters_flower、F_esters_fruity、stale_hops、clove、lactic、acetic、barnyard。它们不输出校准分。'''
    shown=decision[['target','calibration_MAE','prediction_MAE','baseline_MAE','better_folds','adopt']].copy()
    shown['adopt']=shown.adopt.map({True:'采用',False:'未采用'})
    text+='\n\n'+m.table(shown.rename(columns={'target':'维度','calibration_MAE':'映射MAE','prediction_MAE':'预测MAE','baseline_MAE':'基准MAE','better_folds':'改善折数','adopt':'决定'}))
    text+='\n\n配套表提供25维校准面板均值，并以floor(r_hat+0.5)附列最近整数0–6级。此整数规则是透明的暂定输出方式，不是已确认的官方评分或面板多数票。原始记录等权聚合与先按评分者均值聚合的平均差约0.00256分，个别项目差可达0.32038分，聚合口径应向主办方确认。新数据支持部分实质校准，但不代表全部第一问已经解决。'
    i=next(i for i,(h,b) in enumerate(sections) if h.startswith('4 第二问'))
    sections.insert(i,('3.1 新原始评分支持的部分校准',text))
    q=pd.read_csv(HERE/'q3_decision.csv')
    text='''第三问只在原200款内比较v3与两种简单候选：C在每折可信正浓度、单位和变换都匹配的单项坐标上作浓度插值，其他坐标冻结；N冻结五项汇总，采用目标中心距离加改动惩罚，并限制RMS改动≤0.75及单坐标≤2个训练SD。两方案均冻结四缺陷、保留Stale hops弱代理不升高条件；alpha固定0.25、0.5、0.75，不按结果再调参。原生log10插值是浓度的几何平均，不等于浓度线性混配。

预先固定3折、种子42/43/44，共9折。每折全部预处理、供体、目标统计及模型只用该折拟合部分。留出Pils每折3款、留出Tripel为7或8款；实际仅9款不同训练Pils，重复折不能视为27个独立样本。某些其他风格只有2款，三折不能保证每折都有该类，分类诊断受此限制。没有重新使用原50款测试集。

主要证据是到未参加优化的留出Tripel的逐维Wasserstein距离；另报告最近距离、改动、缺失、不可行和回退。两候选都未达到至少5%分布距离改善、7/9折改善及约束/保留不恶化的采纳标准，因此不替换v3。'''
    shown=q[['method','wasserstein','nearest','movement','new_missing','infeasible','fallback']].copy()
    text+='\n\n'+m.table(shown.rename(columns={'method':'方法','wasserstein':'留出分布距离','nearest':'留出最近距离','movement':'改动RMS','new_missing':'新增缺失/款','infeasible':'候选不可行率','fallback':'原谱回退率'}))
    text+='''\n\nC和N的主要分布距离分别比v3差11.34%和7.60%，虽然改动RMS更小，两者仅1/9折优于v3。所有失败候选及全部验证Pils均保留，没有只报成功样本。本次没有证明更好的迁移。

对既有v3的9款训练Pils做固定诊断：供体本身9/9被判Tripel，75%插值后只有2/9被判Tripel，另外7款为Blond；取消四缺陷冻结的描述性对照仍为2/9。因而不能把失败主要归于缺陷冻结，更可能涉及源谱保留、混合谱与多类别决策边界；这是诊断线索，不是因果解释。含原两款历史测试Pils的既有11款结果仍为3/11，没有以它们选择新方案。

9款训练Pils中，DMS、双乙酰、反式-2-壬烯醛各9款有可信正值配对，4VG仅1款；其余4VG零义未知。三次重复比较中可信浓度及有阈值时的超出量均保持不变；反式-2-壬烯醛虽可比较浓度仍不能判阈值安全。这里不能扩展为全部250款或真实感知缺陷保证。

Stale hops另外固定比较训练均值、原森林、岭回归alpha=100，五折MAE为0.3120、0.3127、0.3240，后两者都仅2/5折改善，均未达预定5%且4/5折标准。它是感官描述而非已校准化学指标，未找到可靠化学代理/通用阈值。仍无法保证实际Stale hops不恶化，不为满足措辞升级弱模型。'''
    i=next(i for i,(h,b) in enumerate(sections) if h.startswith('7 结论'))
    sections.insert(i,('6.2 训练内有限改进比较：未采纳新迁移',text))
    sections[-1]=(sections[-1][0],sections[-1][1]+'\n\n[4] Schreurs M, et al. Data from: Predicting and improving complex beer flavor through machine learning. Zenodo version4, 2026-10-24. DOI:10.5281/zenodo.23624915. 新增Raw sensory data anonymized.csv；与旧补充表分开注明版本。https://zenodo.org/records/23624915')
    md='# '+title+'\n\n候选提交版v4：仅新增通过训练内标准的25维校准；第三问保留v3方法。\n\n'+'\n\n'.join('## '+h+'\n\n'+b for h,b in sections)+'\n'
    (OUT/'competition_paper.md').write_text(md,encoding='utf-8');m.pdf(title,sections)
    (OUT/'Q1_SCORING_SCOPE.md').write_text('# v4第一问范围\n\n新公开原始评分经数值指纹配对与交叉复核，38维重建可靠，25维通过预定训练内标准，其中Aroma/Flavour共15维。只对通过维度输出有监督校准面板均值。不是逐人原始评分精确逆变换。\n\n最近整数列仅为floor(r_hat+0.5)暂定规则，官方聚合与整数评分仍待确认。其余维度继续保留标准化预测和相对等级。全部未通过和未复现维度见论文及实验decision表。\n',encoding='utf-8')
    (OUT/'START_HERE.md').write_text('# v4推荐材料\n\n[论文PDF](competition_paper.pdf) · [可编辑稿](competition_paper.md) · [部分校准输出](q1_calibrated_predictions_partial.csv)\n\n[唯一推荐支撑包](submission_candidate_v4.zip)\n\n只采纳25维原始面板均值校准；第三问未发现值得替换v3的新方案。基线计算沿用已验证v3，不冒称全部重新选模。\n',encoding='utf-8')
    print('Built v4:',OUT,flush=True)

def package():
    sys.path.insert(0,str(ROOT/'scripts'))
    from package_competition import check_links
    payload={}
    def put(p,key):payload[key]=p.read_bytes()
    for folder in ['data','frozen_inputs','scripts']:
        names=None
        if folder=='scripts':
            from package_competition import ACTIVE_SCRIPTS
            names=ACTIVE_SCRIPTS
        if folder=='frozen_inputs':names=['source_sha256.json','model_selection_cv_50.csv','test_predictions_50.csv','reference_supplement.xlsx']
        for p in ([ROOT/folder/n for n in names] if names else (ROOT/folder).iterdir()):
            if p.is_file():put(p,p.relative_to(ROOT).as_posix())
    put(ROOT/'requirements_competition.txt','requirements_competition.txt')
    for p in OUT.rglob('*'):
        if p.is_file() and not p.name.startswith('base_') and p.suffix in ['.csv','.json','.pdf','.md','.joblib','.graphml','.png','.svg','.ipynb'] and p.name not in ['package_verification.json','package_execution.json','final_validation_execution.json','post_visual_build.json','history_preservation.json']:
            put(p,'results/'+p.relative_to(OUT).as_posix())
    # Keep experiment evidence and reproducible implementation, omit caches/failed logs/history snapshots.
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ['.py','.csv','.md']:
            put(p,'experiments/limited/'+p.name)
    for n in ['q1_decision.json','q3_fold_scope.json','execution.json']:
        put(HERE/n,'experiments/limited/'+n)
    put(HERE/'sources/raw_sensory_anonymized.csv','experiments/limited/sources/raw_sensory_anonymized.csv')
    payload['README.md']='''# MathorCup v4唯一推荐支撑包

[论文PDF](results/competition_paper.pdf) · [可编辑论文](results/competition_paper.md) · [校准预测](results/q1_calibrated_predictions_partial.csv)

[来源发现](experiments/limited/SOURCE_FINDINGS.md) · [预定方案](experiments/limited/PROTOCOL.md) · [实验结论](experiments/limited/RESULTS.md)

只采纳25维面板平均强度校准，其中Aroma/Flavour15维；不宣称全部原始评分已还原。第三问两种新方案未通过，保留v3方法及全部困难结果。原50款未用于新增方案选择。最近整数只是暂定展示规则。

Python3.13，安装requirements_competition.txt。当前材料验证：
```powershell
python scripts/verify_competition.py --output results
python experiments/limited/verify_experiment.py --output results
```

完整重算（新目录，保护已交付results）：
```powershell
python scripts/run_review.py --output outputs/rebuilt_base --no-package
$env:MATHORCUP_BASE = (Resolve-Path outputs/rebuilt_base).Path
python experiments/limited/run_experiment.py
python experiments/limited/build_v4.py
python experiments/limited/verify_experiment.py --output experiments/limited/adopted_v4
```

中文TrueType字体由本机提供，可设MATHORCUP_CJK_FONT。build_v4复用基线已执行结果并新增校准；请对新生成PDF渲染复核后再运行build_v4.py --package。最终包仅含当前源代码、输入、结果及有限实验，不含v3旧论文或旧ZIP。源数据标识对照为数值证据而非主办方官方映射。AI协助实现，历史完整交互记录未补造；未执行平台提交。
'''.encode('utf-8')
    payload['results/START_HERE.md']=b'# v4\n\n[Paper](competition_paper.pdf) | [Editable paper](competition_paper.md) | [README](../README.md)\n'
    # Existing task alignment is data; avoid old external document links.
    links=check_links(payload)
    manifest={k:hashlib.sha256(v).hexdigest() for k,v in payload.items()}
    archive=OUT/'submission_candidate_v4.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for k,v in payload.items():z.writestr(k,v)
        z.writestr('SUPPORT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    extracted=HERE/'package_check';extracted.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(k)).hexdigest()==h for k,h in manifest.items())
        z.extractall(extracted)
    assert all(hashlib.sha256((extracted/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
    log=[]
    for command in [[sys.executable,'scripts/verify_competition.py','--output','results'],[sys.executable,'experiments/limited/verify_experiment.py','--output','results']]:
        start=datetime.now(timezone.utc).isoformat();begin=time.monotonic()
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env.pop('MATHORCUP_BASE',None)
        p=subprocess.run(command,cwd=extracted,capture_output=True,env=env)
        (HERE/('package_check_'+str(len(log))+'.log')).write_bytes(p.stdout+p.stderr)
        log.append({'command':command[1:],'started_utc':start,'returncode':p.returncode,'seconds':time.monotonic()-begin})
        assert p.returncode==0,p.stderr.decode('utf-8',errors='replace')
    (OUT/'package_verification.json').write_text(json.dumps({'status':'passed','archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(payload),'links':links,'CRC':'passed','file_hashes':'passed','extracted_checks':log},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified v4 package',archive,flush=True)

if __name__=='__main__':
    if '--package' in sys.argv:package()
    else:build()
