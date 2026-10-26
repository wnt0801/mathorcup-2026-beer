"""Build a reproducible paper, figures, review notebooks and compliance inventory.
Run with the same environment as competition_pipeline.py after its stages complete.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import sys
import textwrap

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/competition_completion_v1'

def table(df,columns=None):
    if columns is not None:df=df[columns]
    names=list(df.columns)
    rows=[]
    for row in df.itertuples(index=False,name=None):
        rows.append([f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row])
    return '| '+' | '.join(map(str,names))+' |\n| '+' | '.join(['---']*len(names))+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)


def reference_audit():
    p=OUT/'reference_supplement.xlsx'
    assert hashlib.md5(p.read_bytes()).hexdigest()=='52645e38d90aafda272532e19e2fea1c'
    a=pd.read_csv(ROOT/'data/4.csv').set_index('beer').sort_index()
    b=pd.read_excel(p,sheet_name='Supplementary File S4').set_index('beer').sort_index()
    fields=a.columns.drop(['beer_id','tasting_category_fine'])
    delta=(a[fields]-b[fields]).abs()
    result={'source':'https://zenodo.org/records/10653704','file_url':'https://zenodo.org/records/10653704/files/Supplemental%20Files%20and%20Figure%20source%20files.xlsx?download=1','md5':hashlib.md5(p.read_bytes()).hexdigest(),'sheets':pd.ExcelFile(p).sheet_names,'matched_standardized_targets':len(fields),'maximum_a4_vs_s4_difference':float(delta.max().max()),'conclusion':'S4 is standardized, S3 is a blank form; this supplement does not supply a complete paired original 0–6 training label table or inverse calibration.'}
    (OUT/'reference_provenance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    # Calibrated fields: empirically identify transforms, explicitly retain disagreements.
    x=pd.read_csv(ROOT/'data/1.csv').set_index('beer')
    cal=pd.read_excel(ROOT/'data/8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer').reindex(x.index)
    records=[]
    for f in cal.columns.drop(['beer_id','tasting_category_fine']):
        mask=x[f].notna()&cal[f].notna()&cal[f].gt(0)
        for name,v in [('identity',cal[f]),('log10',np.log10(cal[f].where(cal[f]>0))),('ln',np.log(cal[f].where(cal[f]>0)))]:
            err=(x.loc[mask,f]-v.loc[mask]).abs()
            records.append(dict(field=f,candidate_transform=name,n=int(mask.sum()),matched=int(err.le(1e-6).sum()),max_error=err.max(),a1_missing=int(x[f].isna().sum()),a8_missing=int(cal[f].isna().sum()),a8_zero=int(cal[f].eq(0).sum())))
    pd.DataFrame(records).to_csv(OUT/'calibrated_fields_transform_audit.csv',index=False,encoding='utf-8-sig')


def figures():
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    import networkx as nx
    font=Path('C:/Windows/Fonts/simhei.ttf')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'savefig.dpi':220,'axes.spines.top':False,'axes.spines.right':False})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    def export(fig,name):
        fig.savefig(figdir/(name+'.png'),bbox_inches='tight');fig.savefig(figdir/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    m=pd.read_csv(OUT/'q1_metrics.csv').query("split=='heldout_test'").sort_values('target')
    fig,ax=plt.subplots(figsize=(9,12));pos=np.arange(len(m));ax.barh(pos-.18,m.baseline_MAE,height=.35,label='训练均值基准',color='#b8c2ca');ax.barh(pos+.18,m.MAE,height=.35,label='冻结模型',color='#216869');ax.set_yticks(pos,m.target,fontsize=8);ax.invert_yaxis();ax.set_xlabel('测试MAE（附件4标准化尺度）');ax.legend();ax.set_title('50项感官预测：测试结果复现');fig.tight_layout();export(fig,'01_prediction_errors')
    d=pd.read_csv(OUT/'q1_missingness_sensitivity.csv').groupby('additional_missing_rate').MAE.agg(['mean','std'])
    fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(d.index*100,d['mean'],yerr=d['std'],marker='o',capsize=4,color='#216869');ax.set_xlabel('测试输入额外随机遮蔽比例（%）');ax.set_ylabel('50维平均MAE');ax.set_title('冻结模型的缺失敏感性（每档5次）');fig.tight_layout();export(fig,'02_missingness')
    g=nx.read_graphml(OUT/'q2_official_compound_graph.graphml');nodes=[n for n in g if g.degree(n)>0];g=g.subgraph(nodes)
    fig,(ax,hubax)=plt.subplots(1,2,figsize=(11,6),gridspec_kw={'width_ratios':[1.25,1]})
    pos=nx.spring_layout(g,seed=42,iterations=80,weight='weight')
    nx.draw_networkx_edges(g,pos,ax=ax,width=.7,alpha=.45,edge_color=['#216869' if g[u][v]['rho']>0 else '#c65b45' for u,v in g.edges])
    nx.draw_networkx_nodes(g,pos,ax=ax,node_size=[20+10*g.degree(n) for n in g],node_color='#e7ad4b',alpha=.9)
    hubs=sorted(g,key=lambda n:g.degree(n),reverse=True)[:12]
    hubax.barh(np.arange(len(hubs)),[g.degree(n) for n in hubs],color='#216869')
    hubax.set_yticks(np.arange(len(hubs)),hubs,fontsize=8);hubax.invert_yaxis();hubax.set_xlabel('连接数');hubax.set_title('连接数最高的12项',fontsize=11)
    ax.set_title('非孤立节点与共变结构',fontsize=11);ax.axis('off')
    fig.suptitle('附件2化合物网络：|相关|≥0.6',fontsize=13);fig.tight_layout();export(fig,'03_compound_network')
    raw=pd.read_csv(ROOT/'data/1.csv').set_index('beer');s=pd.read_csv(OUT/'split_manifest.csv').set_index('beer');features=raw.columns.drop(['beer_id','tasting_category_fine']);fit=raw.loc[s.index[s['split'].eq('training')],features];im=SimpleImputer(strategy='median').fit(fit);sc=StandardScaler().fit(im.transform(fit));pca=PCA(2,random_state=42).fit(sc.transform(im.transform(fit)))
    fake=pd.read_csv(OUT/'q3_generated_native_spectra.csv').set_index('sample_id');line=pd.read_csv(OUT/'q3_generation_lineage.csv').set_index('sample_id');fig,ax=plt.subplots(figsize=(8,5))
    for style,color in zip(['Tripel','Kriek','Lambic'],['#216869','#b84259','#d59b2b']):
        r=fit.loc[raw.loc[fit.index,'tasting_category_fine'].eq(style)];f=fake.loc[line['style'].eq(style)];rz=pca.transform(sc.transform(im.transform(r)));fz=pca.transform(sc.transform(im.transform(f)));ax.scatter(fz[:,0],fz[:,1],s=12,alpha=.2,color=color);ax.scatter(rz[:,0],rz[:,1],s=45,marker='x',color=color,label=style+'训练真实')
    ax.set_xlabel('PC1（训练集拟合）');ax.set_ylabel('PC2');ax.set_title('条件生成谱：圆点为生成，叉号为训练真实');ax.legend(fontsize=8);fig.tight_layout();export(fig,'04_conditional_generation')
    prof=pd.read_csv(OUT/'q3_style_sensory_profiles.csv');targets=['A_malt_all','A_hops_all','A_esters_all','F_malt_all','F_hops_all','F_esters_all','acidity','bitternes','sweetness','body','alcohol','overall'];mat=prof.pivot(index='style',columns='target',values='predicted_mean').loc[['Tripel','Kriek','Lambic'],targets];fig,ax=plt.subplots(figsize=(10,3));implot=ax.imshow(mat,cmap='RdBu_r',vmin=-1.5,vmax=1.5,aspect='auto');ax.set_xticks(np.arange(len(targets)),targets,rotation=40,ha='right',fontsize=8);ax.set_yticks(np.arange(3),mat.index)
    for i in range(3):
        for j in range(len(targets)):ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(implot,ax=ax,label='预测标准化得分');ax.set_title('三种风格的生成感官画像（非原始0–6分）');fig.tight_layout();export(fig,'05_style_profiles')
    tr=pd.read_csv(OUT/'q3_transfer_summary.csv');fig,ax=plt.subplots(figsize=(8,4));p=np.arange(len(tr));ax.bar(p-.18,tr.source_target_distance,width=.36,label='迁移前',color='#b8c2ca');ax.bar(p+.18,tr.transferred_target_distance,width=.36,label='迁移后',color='#216869');ax.set_xticks(p,tr.source_beer);ax.set_xlabel('Pils/Lager beer编号');ax.set_ylabel('到Tripel训练谱的最小RMS距离');ax.set_title('带缺陷约束的跨风格迁移');ax.legend();fig.tight_layout();export(fig,'06_transfer')


def notebooks():
    import nbformat as nbf
    from nbclient import NotebookClient
    names=[('06_competition_scoring_review.ipynb','问题1：冻结模型复现、缺失敏感性与相对分级',[
        ('注意','本笔记本读取新生成的独立结果。真实原始0–6分逆变换仍不可识别；所有等级列都是训练集分位数相对等级。训练OOF成绩并非无偏的嵌套选模估计，因为候选选择曾使用同一训练集CV。'),
        ('code',"m=pd.read_csv(OUT/'q1_metrics.csv'); display(m.groupby('split').mean(numeric_only=True)); display(pd.read_csv(OUT/'q1_train_quantile_cutpoints.csv')); display(pd.read_csv(OUT/'q1_missingness_sensitivity.csv').groupby('additional_missing_rate').MAE.agg(['mean','std'])); display(pd.read_csv(OUT/'barcode_precision_audit.csv').head())"),
        ('code',"p=pd.read_csv(OUT/'q1_out_of_sample_predictions.csv'); assert len(p)==12500; assert not p.duplicated(['beer','target']).any(); display(p.head(20)); display(Image(filename=str(OUT/'figures/01_prediction_errors.png')))"),
    ]),('07_competition_network_all50.ipynb','问题2：官方相关矩阵、全部50维Top-3与化学证据',[
        ('说明','附件2实际匹配Spearman而非题面所称Pearson；选取218项单项测量候选，其他指标不作为“化合物”。bootstrap=200，频率衡量排名稳定性，不代表因果概率。全部50维输出，超出Aroma/Flavour的20维作补充。'),
        ('code',"display(pd.read_csv(OUT/'q2_official_correlation_type_check.csv')); e=pd.read_csv(OUT/'q2_top3_all50_evidence.csv'); assert len(e)==150 and e.groupby('target').size().eq(3).all(); display(e); display(pd.read_csv(OUT/'q2_method_sensitivity.csv'))"),
        ('code',"display(Image(filename=str(OUT/'figures/03_compound_network.png')))"),
    ]),('08_competition_conditional_generation.ipynb','问题3：条件生成、风味指纹与受约束迁移',[
        ('说明','按训练集风格条件进行近邻谱插值，参数规则固定，不基于测试成绩选取。输出是附件1混合尺度上的统计化合物谱，不是已验证浓度配方或酿造工艺。插值不自动保证汇总指标恒等式。四个缺陷原生字段完全冻结；未知值保持未知；Stale hops仅有模型代理约束，非实测缺陷保证。'),
        ('code',"v=pd.read_csv(OUT/'q3_generation_heldout_validation.csv'); display(v.groupby('mode').mean(numeric_only=True)); display(v); display(pd.read_csv(OUT/'q3_style_diagnostic.csv')); display(pd.read_csv(OUT/'q3_style_sensory_profiles.csv').query(\"style in ['Tripel','Kriek','Lambic']\"))"),
        ('code',"t=pd.read_csv(OUT/'q3_transfer_summary.csv'); display(t); assert t.defect_native_exactly_preserved.all(); assert (t.transferred_stale_auxiliary_guard<=t.source_stale_auxiliary_guard+1e-12).all(); display(pd.read_csv(OUT/'q3_transfer_defect_constraints.csv')); display(Image(filename=str(OUT/'figures/06_transfer.png')))"),
        ('code',"display(Image(filename=str(OUT/'figures/04_conditional_generation.png'))); display(Image(filename=str(OUT/'figures/05_style_profiles.png')))"),
        ('复现','项目根目录执行：python scripts/competition_pipeline.py --stage all。随后执行 python scripts/build_competition_deliverables.py。模型载入前将 scripts 加入sys.path；只读取自己生成的joblib文件。')
    ])]
    for name,title,blocks in names:
        p=ROOT/'notebooks'/name
        if p.exists():raise FileExistsError(p)
        nb=nbf.v4.new_notebook();nb.cells=[nbf.v4.new_markdown_cell('# '+title),nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd() if (Path.cwd()/'data').exists() else Path.cwd().parent\nOUT=ROOT/'outputs/competition_completion_v1'\npd.set_option('display.max_rows',160)\npd.set_option('display.max_columns',25)")]
        for kind,text in blocks:nb.cells.append(nbf.v4.new_code_cell(text) if kind=='code' else nbf.v4.new_markdown_cell('## '+kind+'\n\n'+text))
        nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
        NotebookClient(nb,timeout=120,resources={'metadata':{'path':str(ROOT)}}).execute();nbf.write(nb,p);print('executed notebook',name,flush=True)


def report():
    m=pd.read_csv(OUT/'q1_metrics.csv');held=m.query("split=='heldout_test'");oof=m.query("split=='training_oof'")
    top=pd.read_csv(OUT/'q2_top3_all50_evidence.csv');val=pd.read_csv(OUT/'q3_generation_heldout_validation.csv');v=val.groupby('mode').mean(numeric_only=True)
    transfer=pd.read_csv(OUT/'q3_transfer_summary.csv');diag=pd.read_csv(OUT/'q3_style_diagnostic.csv').iloc[0];nodes=pd.read_csv(OUT/'q2_graph_nodes.csv');edges=pd.read_csv(OUT/'q2_official_graph_edges.csv')
    guardcv=pd.read_csv(OUT/'q3_stale_guard_training_cv.csv')
    missing=pd.read_csv(OUT/'q1_missingness_sensitivity.csv').groupby('additional_missing_rate').MAE.agg(['mean','std']).reset_index()
    stable=int(top.bootstrap_frequency.ge(.8).sum());weak=int(top.bootstrap_frequency.lt(.5).sum());good=int(transfer.status.eq('feasible_improvement').sum())
    sections=[]
    def add(title,body): sections.append((title,textwrap.dedent(body).strip()))
    add('摘要',f'''
    针对啤酒化合物与感官数据混合尺度、真实缺失和样本较少的问题，建立“尺度核验—冻结预测—关联证据—条件谱生成—约束迁移”的分析流程。实际附件包含250款啤酒、231个化学/理化输入和50个感官目标，与题面若干描述存在差异，本文按文件证据处理。
    问题1沿用既有训练集交叉验证选择的模型，逐项复现原测试预测。50维平均测试MAE由{held.baseline_MAE.mean():.4f}降至{held.MAE.mean():.4f}，{int(held.MAE.lt(held.baseline_MAE).sum())}维优于均值基准。另给出12500条逐啤酒逐维的样本外预测记录与训练分位数相对0–6级。由于原始评分和逆变换参数未提供，不能将相对等级认定为原始0–6分恢复。
    问题2发现附件2与重算Spearman矩阵的最大差异仅6.66×10^-16。采用218个单项测量候选构建网络（{len(edges)}条边），对全部50维给出150条Top-3证据，200轮bootstrap中{stable}条入选频率不低于80%。同时报告风格调整、来源与气味证据，避免把相关当作因果。
    问题3以训练样本构成风格条件局部生成分布，为22种风格各生成100条谱及对应感官画像。在11款Pils/Lager向Tripel的约束迁移中，{good}款找到目标距离下降的可行候选，四个缺陷原生字段保持不变，Stale hops辅助代理不升高。生成谱是混合尺度下的统计候选，尚未经过酿造或盲评验证；缺失阈值和原始评分校准仍构成提交限制。
    关键词：啤酒风味；尺度核验；相对等级；Spearman网络；条件生成；约束迁移。
    ''')
    add('1 问题分析与任务口径','''
    第一问要求以附件1给出附件3定义的Aroma与Flavour维度0–6分级。实际评分卡包含Aroma 15维、Flavour 15维，另有20个感官项目，合计50维。本文核心分析覆盖30个A/F目标，并将全50维结果作为完整补充。题面“约40维”不能用来随意删除或增造维度。
    第二问要求利用附件2建立化合物网络，并给出每个感官维度的Top-3与稳定性、化学合理性。统计关联能发现候选因素，但同源代谢和风格均可能造成混杂；关键因素在本文指关联候选，不等于已经证实的单体作用。
    第三问要求按tasting_category_fine生成合理化合物谱、感官画像和跨风格迁移，并保持缺陷不恶化。样本分布不均，风格仅3–31款，因此选择可审计的小样本局部生成分布，给出分布与多样性诊断。谱生成、物理浓度配方和实际酿造可行性是不同证据层次。
    赛题另要求完整论文、清晰量化图表、可复现源码。本文保留原有文件，所有新产物存入独立目录。未向赛事平台提交；队伍身份和赛事额外格式规范不在当前文件中，需正式提交前核对。
    ''')
    add('2 数据结构、冲突与来源核验','''
    附件1为250×234，去除beer、beer_id和风格后有231个输入；附件4为250×53，含50个标准化连续目标。附件3有beer_1至beer_12和summary共13个sheet，而非题面所述10个。评分模板中的0和公式缓存不是250款啤酒的实际人工评分，禁止据此构造监督标签。附件6实际为44个化学/理化指标×50感官项目，而非题面46；附件7为250×189，去除3列标识后有186个属性，与题面187也不同。
    以唯一beer编号对齐，两个表的编号集合与风格逐项一致。附件4的beer_id以低精度科学计数法导出，部分条码数值已损失精度，单独记录到barcode_precision_audit.csv，不以条码猜测修复编号。原有数据、Notebook、输出均保存SHA256清单。
    四种重点化合物采用独立核验Notebook的结论。附件8声明DMS单位ug/L；双乙酰、4VG、反式-2-壬烯醛单位mg/L。DMS与4VG的附件8正浓度分别249和82条，在附件1中符合log10，误差不超过5×10^-10。两列分别1和168个零值，在附件1对应固定有限值，逆log10约为最小正浓度的一半。这支持某种零值替代预处理，但不能证明零值是绝对零、未检出或缺失。
    双乙酰241条共同非缺失记录中238条原值对应；beer156、34、151分别为附件1/8的0.434130/0.123114、0.289622/0.530466、0.139752/0.043173。另9个附件1缺失位置在附件8均为0，本文视为来源冲突而非已知零。beer73在两附件均约121.731018 mg/L，且源格为数字；一致不能证明测量真实性，因此不删行、不改单位、不缩尾原数据。既有岭回归内部的训练分位数裁剪属于模型预处理，不改变原文件。
    反式-2-壬烯醛的249条共同有效值中247条原值匹配；beer158、130数值有差异，177在附件8缺失。其阈值mean与median均缺失。附件8三项既有19、0.09、0.20分别是DMS、双乙酰、4VG的threshold_median，而不是本项目实验确证的风格特异缺陷界限。Stale hops是感官描述，附件中没有同名的单一化合物浓度阈值。
    进一步下载原研究Zenodo v3公开补充表，MD5与站点声明52645e38d90aafda272532e19e2fea1c相同，完整列出29个sheet。其S4仍为同一类标准化评分，S3仍为空白表；其他验证实验为额外处理条件，不能充当全部250款原始配对标签。公开补充资料未解决完整原始0–6标签和逆变换校准问题。原研究确实说明感官分析使用z分数及部分化学量使用对数变换[2–3]。
    ''')
    add('3 问题1：冻结模型与相对分级','''
    3.1 连续预测。沿用random_state=42、按22种风格分层的200/50训练/测试划分，复用原model_selection_cv_50.csv中已冻结的逐目标选择。候选为训练均值基准、带中位数填补和标准化的岭回归、随机森林。既有岭回归先按训练折1%与99%分位数裁剪，alpha=100；森林为200棵树、max_depth=6、min_samples_leaf=5、max_features=0.7。50维选择为42个森林、5个岭回归、3个均值基准，不用新的测试结果继续选模型。
    缺失仅在模型输入的副本中按训练子集的列中位数填补，原始数据不改写、不把缺失当0。裁剪界点、填补值、标准化参数均只由相应拟合子集计算。先重建200样本模型，对原50个测试beer逐项对照既有预测文件，允许误差1e-8；全部通过。
    3.2 原始评分不可识别。即使假设每个属性满足z=(r-mu)/sigma，未给出mu与sigma也不能唯一反求r。对于当前标准化数值范围，r=3+0.4z与r=2+0.5z都可落入0–6，但彼此不同；假定离散、对个体或对面板汇总标准化，又会带来更多不可逆信息。把观察min/max映射到0/6、整体加常数或四舍五入均不能成为真实还原证据。
    3.3 可复现的替代输出。对每个目标在训练子集计算六个分位点q_j=Q(j/7)，j=1,…,6；定义g(z)=sum_j I(z≥q_j)，输出0至6相对等级。等级同分并非原始强度相同，不同目标的等级也不能充当绝对感官单位。每个训练OOF样本的界点仅用该折拟合子集计算，测试界点用全部200个训练样本计算。所有维度的全训练界点均互异；使用side=right处理边界同分。
    3.4 评估与局限。训练部分采用5折样本外预测，测试50款复用冻结模型。训练OOF不是无偏的嵌套选择评估，因为原模型选择用过同一训练集；据此得出的残差区间只能作描述性不确定性指示，不能宣称严格90%覆盖保证。区间半宽取OOF绝对残差的90%高阶分位数，不用测试残差调区间。
    ''')
    metrics=m.groupby('split')[['MAE','baseline_MAE','R2','relative_grade_MAE','relative_grade_accuracy','interval_coverage']].mean().reset_index()
    metrics['split']=metrics['split'].map({'heldout_test':'测试','training_oof':'训练OOF'})
    metrics=metrics.rename(columns={'split':'数据','baseline_MAE':'基准MAE','relative_grade_MAE':'等级MAE','relative_grade_accuracy':'分级准确率','interval_coverage':'区间覆盖率'})
    missing=missing.rename(columns={'additional_missing_rate':'额外遮蔽率','mean':'平均MAE','std':'MAE标准差'})
    add('4 问题1实验结果',f'''
    {table(metrics)}

    测试连续MAE改善{(1-held.MAE.mean()/held.baseline_MAE.mean())*100:.2f}%，42/50维改善。相对分级测试MAE为{held.relative_grade_MAE.mean():.3f}级，基准为{held.relative_baseline_grade_MAE.mean():.3f}级；平均准确率约{held.relative_grade_accuracy.mean()*100:.2f}%，表明七档精细区分仍然困难，不应只展示连续MAE而回避分级误差。50维MAE等权平均，不能解释为所有单项均显著改善。

    ![图1：冻结模型的逐维测试误差](figures/01_prediction_errors.png)

    在测试输入额外独立遮蔽0%、5%、10%、20%观测值，每档5次，模型与预处理固定；结果如下。该试验是MCAR型人为遮蔽，不代表真实非随机缺失，且用作诊断而非据此重新选模。

    {table(missing)}

    ![图2：缺失敏感性](figures/02_missingness.png)
    ''')
    add('5 问题2：化合物网络与稳定因素',f'''
    按化学字典将231项输入分为单项测量候选、汇总值、理化测量和混合定义待核实项。Top-3仅从218项单项候选中选，避免总酯等汇总量冒充一个分子。附件2经名称映射与希腊字母乱码修正后完整一一对应231项，未随意近似匹配。
    对26565个不同指标对，附件2与1.csv重算Spearman的平均绝对差1.59×10^-16、最大6.66×10^-16；与Pearson平均差0.08768，最大1.12343。原研究公开绘图代码也以Spearman计算化学相关[4]。故保留附件2实际语义，并将Pearson作为敏感性比较，而不按错误文字给现成矩阵重新命名。
    无向网络以218个候选为节点，|rho|≥0.6为边；得到{len(edges)}条边、{int(nodes.degree.gt(0).sum())}个非孤立节点。既有100轮网络bootstrap以相关强度及同号保留率筛选得到172条稳定边，作为边级稳定性补充。连接多只表示共变，不表示化学反应方向或生化通路。
    对每个感官目标，成对使用双方均非缺失的样本，要求有效数≥200，按Spearman绝对值取前三。对整行beer重采样200轮，每轮在全218候选中重新排名，定义入选频率f_j=200轮进入Top-3的次数/200。附表另给已选候选相关系数的百分位bootstrap区间，这些区间未校正事后选择，不作为确认性显著性结论。
    全50维共150条候选，其中{stable}条入选频率≥0.8、{weak}条低于0.5，提示不少维度的前三排序不稳定。对每对变量的秩值减去各风格组均值，再计算残差相关；{int(top.direction_agrees.sum())}/150条方向一致，其余需要特别警惕风格混杂。这种调整仅移除风格组平均秩差异，不能控制未测混杂。
    ![图3：附件2化合物共变网络](figures/03_compound_network.png)
    ''')
    example=top[top.target.isin(['A_malt_all','A_hops_all','A_esters_all'])][['target','rank','chemical_name','rho','bootstrap_frequency','style_adjusted_rho']]
    add('6 化学合理性与反证',f'''
    {table(example)}

    化学论证以附件8的origin_paper、off_flavors和unit为证据，不把名称相似当作已证实机制。麦芽香气前三包含甲基吡嗪（Malt来源），与麦芽相关是有来源支持的候选；乙醛、乙醇属Yeast来源，可能作为发酵或风格共同变化的代理。尤其乙醇与整体麦芽香气的相关由约0.335变为风格调整后约-0.103，不能宣称增加乙醇会增加麦芽香。
    整体酯香中的乙酸乙酯有酯类类别与酵母来源支持；乙醛与正己醇则不应仅因统计排名高就称为“酯香因果物质”。酒花整体香气前三中的辛酸乙酯、癸酸乙酯同属酵母相关酯，反式-2-壬烯醛在附件标注为Others且随陈化增加；它们与酒花香气的相关更需要通过风格内与共变网络解释，而不能据此建议添加陈化醛。原研究也提示共线化合物可相互代替成为重要预测因子[2]。
    完整q2_top3_all50_evidence.csv逐项列出150条候选的来源、气味记录、单位、原始与调整相关、稳定性和区间。气味字典为空表示没有提供该条文字证据，不自动补造香气描述。论文附录列出全50维前三，供读者逐项复核。
    ''')
    add('7 问题3：风格条件生成模型','''
    7.1 模型。令风格s的训练集为D_s。在训练集列中位数填补、StandardScaler变换后的空间中，建立每个样本的同风格最近邻集合N_k(i)，k=min(5,n_s-1)。先均匀抽锚点i，再均匀抽其近邻j，采样lambda~Beta(1,4)，生成x_tilde=(1-lambda)x_i+lambda x_j。离散风格条件与连续局部核共同定义p_hat(x|s)。不是声称训练了VAE或GAN；这是可审计的经验条件生成模型。
    缺失仅在距离计算副本中填补。生成的原生谱若某输入在任一父样本缺失，则该坐标保持缺失，并保存父beer和lambda。完整谱可由训练父样本与lineage重建。已观测坐标位于父样本区间，故不会超出全训练范围；原生非负物理参数也不会因线性插值产生负值。log10坐标上的插值对应浓度的几何路径，原值坐标对应算术路径。对变换尚未核实的字段只声明原生尺度，不自动逆变换。
    这项构造限制外推，但也会压缩方差、遗漏稀有模态。汇总指标同样做统计插值，尚未证明满足物理守恒或汇总恒等式，因此输出不能直接称为可投产配方。原生谱与真实浓度、合法配料、发酵工艺之间仍需实验校验。
    7.2 固定验证。生成规则在看验证数值前固定，训练仅使用原200款；一次性在50款留出样本上比较条件局部生成、同风格直接重采样、无条件局部生成。后两者是基线，不用于测试调参。每风格至少一个留出样本，部分只有1–2个，因此按风格等权的统计很不稳定，不作显著性优越宣称。
    指标包括训练尺度的逐维一维Wasserstein距离、均值RMS偏移、生成点到训练谱最近距离、精确重复比例、训练范围外比例与缺失率。网络保真度以既有稳定边在各风格真实训练谱与生成谱中的Spearman差异衡量；它是拟合检查，不是独立泛化证明。单独固定一个风格分类器作诊断，其测试准确率约60%，说明类别本就重叠；生成条件一致率不能等同感官真实性。
    ''')
    vtable=v[['marginal_wasserstein','generated_nearest_train','duplicate_fraction','outside_training_range','generated_missing_rate']].reset_index().rename(columns={'mode':'方法','marginal_wasserstein':'边际W距离','generated_nearest_train':'最近训练距离','duplicate_fraction':'精确重复率','outside_training_range':'训练范围外比例','generated_missing_rate':'生成缺失率'})
    vtable['方法']=vtable['方法'].map({'conditional_local':'条件局部','conditional_resample':'条件重采样','unconditional_local':'无条件局部'})
    fidelity=pd.read_csv(OUT/'q3_generated_network_fidelity.csv').groupby('style').absolute_difference.mean().reset_index()
    add('8 条件生成实验结果',f'''
    共为22种风格各生成100条，即2200条231维原生谱；每条配50维冻结模型预测与相对分级，保留原生坐标、谱血缘和预测输出，不添加伪造的样本实测标签。

    {table(vtable)}

    条件生成的逐维Wasserstein均值为{v.loc['conditional_local','marginal_wasserstein']:.4f}，无条件基线为{v.loc['unconditional_local','marginal_wasserstein']:.4f}。直接重采样精确重复率100%，局部模型为0%；但局部生成到训练集平均最近距离仅{v.loc['conditional_local','generated_nearest_train']:.4f}，明显低于真实留出样本的{v.loc['conditional_local','real_nearest_train']:.4f}，提示覆盖不足与近训练集生成，而不是充分学习整个风格分布。
    固定风格诊断器的留出准确率{diag.heldout_style_accuracy*100:.1f}%，平衡准确率{diag.heldout_balanced_accuracy*100:.1f}%；生成谱条件一致率{diag.generated_condition_agreement*100:.2f}%。后者接近100%部分源于局部生成与训练谱接近，不证明新颖风格创造成功。Tripel、Kriek、Lambic的网络边相关平均差如下。

    {table(fidelity)}

    ![图4：条件生成化学谱的训练PCA投影](figures/04_conditional_generation.png)

    ![图5：Tripel、Kriek、Lambic生成感官画像](figures/05_style_profiles.png)

    q3_style_sensory_profiles.csv给出每个风格、每个目标的预测均值及5%–95%生成分位数。该区间表示模型内生成变化，不含完整预测误差，也不是95%置信区间。图中的酸度、酒花或酯香差异是标准化预测，不能标记为真实0–6分。
    ''')
    add('9 跨风格迁移与缺陷约束',f'''
    对11款Pils/Lager样本，将目标锚点限制为训练集Tripel谱，枚举alpha∈{{0.25,0.5,0.75}}的原生插值，并加入alpha=0原谱候选。上限0.75用于至少保留25%源坐标，避免完全替换为目标供体。四个缺陷坐标DMS、双乙酰、4VG、反式-2-壬烯醛在插值后恢复为源值，包括缺失状态。
    在可行候选中最小化“到最近Tripel训练谱的标准化RMS距离+0.1×改动RMS”。系数是明确的设计取值，未用测试评分调节。源缺失值不凭空补全，因此仍有部分坐标未知。全部候选、可行性、目标距离与最后选择保存在search_frontier表，不仅展示成功样本。
    对四项已知缺陷坐标，原值完全保持是比固定上阈值不增更强的单调约束：若同一可靠单调逆变换成立，则任意同一阈值下max(C-tau,0)不变。它不使已有超阈值缺陷消失，也不能证明未知缺失安全。对于双乙酰来源冲突，本次选择题面指定的附件1作为生成主坐标，附件8只作独立校验，不覆盖冲突。迁移保持的是附件1测量表征，不是在声称解决了实验真值。
    原冻结Stale hops预测器为常数均值模型，其不升高约束本身没有区分力。为避免空约束，另用预先固定的森林配置只在200款训练数据拟合辅助Stale hops代理，并要求其预测不高于源谱。这不是根据测试成绩重选第一问模型，原模型与其结果保持不变。辅助代理5折训练CV平均MAE为{guardcv.MAE.mean():.4f}，均值基准为{guardcv.baseline_MAE.mean():.4f}，平均R2为{guardcv.R2.mean():.4f}，表现较弱。因而只能称“代理约束满足”，不称“陈旧酒花缺陷实测不恶化”；新增非恒定约束解决空约束问题，未解决预测可信度问题。附件没有该描述的单一浓度阈值，也不编造。
    {good}/11款找到可行距离改进；平均相对距离改善{transfer.relative_distance_improvement.mean()*100:.2f}%。训练源样本与留出源样本在summary中分开标注，所有目标供体均来自训练集。该结果是对固定代理和距离目标的优化演示，不是11项独立生产验证，也不把优化后的模型预测当成实验结果。

    {table(transfer[['source_beer','alpha','relative_distance_improvement','source_stale_auxiliary_guard','transferred_stale_auxiliary_guard']].rename(columns={'source_beer':'源beer','alpha':'迁移强度','relative_distance_improvement':'距离相对改善','source_stale_auxiliary_guard':'Stale代理前','transferred_stale_auxiliary_guard':'Stale代理后'}))}

    ![图6：迁移前后目标风格距离](figures/06_transfer.png)
    ''')
    add('10 鲁棒性、可复现性与局限','''
    第一，所有源文件建立哈希清单，旧Notebook、CSV与GraphML均保留。关键标识使用beer而非顺序或精度损失的条码。独立核验Notebook保留四种化合物的原始Excel格地址、类型、格式和批注检查；本次新增材料不篡改该历史记录。
    第二，冻结原模型选择；训练OOF与原测试结果清晰分开。该测试集在历史工作中已被查看，当前复现不是一批全新盲测，不能夸大其独立性。新增缺失试验不用于模型挑选。生成规则固定后报告三种方法结果，不因为留出表现继续挑选生成超参数。迁移辅助代理的设计针对常数约束这一结构性缺陷，不以留出感官成绩选择。
    第三，Top-3稳定性在全部候选中重排名，避免只对原始前三做自我确认。风格调整提供混杂敏感性，Pearson与Spearman的Top-3 Jaccard表提供统计定义变化下的敏感性。这里只是探索性分析；未做独立新样本验证、未控制多重选择后的推断误差。
    第四，生成器受限于小样本、同风格支持和局部核，范围内不等于化学合理。PCA二维图也不能证明高维分布等价。近训练距离、相关差异和缺失率均如实报告，不能仅凭漂亮风味指纹宣称优越。需要以新啤酒、重复检测与盲评检验泛化。
    第五，真正满足原始0–6分要求仍需原始面板标签、对应标准化方式及参数，或组织方批准相对等级口径。缺陷阈值需明确检测单位、风格背景与实验依据；反式-2-壬烯醛、Stale hops的阈值未解决。本论文给出的保守不增设计缓解阈值缺失，但不取代完整缺陷判定。
    ''')
    add('11 结论与建议',f'''
    本项目已经形成可复现的三问研究实现：50维标准化预测复现、12500条样本外相对分级记录、150条带稳定性与来源的Top-3证据、官方矩阵网络、2200条风格条件生成谱及11款跨风格迁移。测试MAE为{held.MAE.mean():.4f}，相较均值基准改善约18%；化合物关联中只有{stable}/150条达到80%入选稳定性，不能笼统宣称所有前三可靠。
    优先建议补齐原始评分校准和有冲突的化合物实验记录；针对高值与零值做来源复核，而不是删行或按猜测换单位。在可确定单位的化合物上开展定量检测，建立风格特异感知/缺陷实验；最后以盲评和小试酿造验证生成与迁移。当前交付可作为完整研究稿和支撑材料，但不应标记为“全部硬要求已验证满足”的最终提交版。
    ''')
    add('参考文献与数据来源','''
    [1] 2026年MathorCup数学应用挑战赛大数据竞赛，赛道A初赛题面，项目data目录PDF，3页；附件1–8及数据说明。
    [2] Schreurs M, Piampongsant S, Roncoroni M, et al. Predicting and improving complex beer flavor through machine learning. Nature Communications, 2024, 15:2368. DOI:10.1038/s41467-024-46346-0. https://pmc.ncbi.nlm.nih.gov/articles/PMC10966102/
    [3] Schreurs M, et al. Data from: Predicting and improving complex beer flavor through machine learning, Zenodo v3. DOI:10.5281/zenodo.10653704. https://zenodo.org/records/10653704 。下载文件名Supplemental Files and Figure source files.xlsx，MD5见来源核验记录。
    [4] 同一Zenodo记录的Main Figure Generator.Rmd，化学相关表生成部分，方法明确为Spearman。https://zenodo.org/records/10653704/files/Main%20Figure%20Generator.Rmd?download=1
    访问日期：2026-10-26。本文未导入Kaggle数据、未执行外部作者脚本、未载入外部预训练pickle，外部资料仅用于溯源。
    ''')
    add('附录A 全部50维Top-3与稳定性','以下数值为200轮入选频率，不是因果置信概率。完整来源和调整相关见配套CSV。')
    appendix=[]
    for t,g in top.groupby('target',sort=False):appendix.append({'target':t,**{f'Top{int(r["rank"])}':f'{r.chemical_name} ({r.bootstrap_frequency:.2f})' for _,r in g.iterrows()}})
    sections[-1]=(sections[-1][0],sections[-1][1]+'\n\n'+table(pd.DataFrame(appendix)))
    add('附录B 复现与AI使用说明','''
    在项目根目录，安装requirements_competition.txt后执行：python scripts/competition_pipeline.py --stage all；再执行python scripts/build_competition_deliverables.py。原有CSV不会被写入，新增产物集中在outputs/competition_completion_v1。运行脚本会验证原文件哈希，并在复现冻结预测不一致时直接报错。
    06、07、08三个新增Notebook均已执行并保存输出，分别对应预测复查、50维关联证据、条件生成与迁移。它们读取新结果，完整计算由脚本承担，避免手工复制和依赖历史交互变量。模型文件载入前将scripts加入sys.path；不要加载不可信外部joblib。
    AI参与赛题提取、代码编写、材料组织及解释草拟；所有报告数值来自实际程序输出，旧结果用于冻结复现。AI提出的代理方法、推断与实验未知项已分开标注；队伍需按赛事最新AI规范核对使用声明和对话记录要求。该记录描述实际使用，不冒充人工独立完成。
    ''')
    title='混合尺度啤酒数据的感官预测、化合物网络与受约束风味谱生成'
    md='# '+title+'\n\n研究稿 v1；真实0–6分与部分缺陷阈值尚待补证，非已核准最终提交版。\n\n'+'\n\n'.join('## '+t+'\n\n'+b for t,b in sections)+'\n'
    (OUT/'competition_paper.md').write_text(md,encoding='utf-8')
    return title,sections


def pdf(title,sections):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4
    pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/simhei.ttf'))
    styles=getSampleStyleSheet()
    body=ParagraphStyle('CNBody',fontName='CN',fontSize=9.2,leading=15.5,spaceAfter=7,wordWrap='CJK')
    head=ParagraphStyle('CNHead',parent=body,fontSize=14,leading=20,spaceBefore=14,spaceAfter=9,keepWithNext=True,textColor=colors.HexColor('#164748'))
    small=ParagraphStyle('CNSmall',parent=body,fontSize=7.2,leading=11,spaceAfter=0)
    caption=ParagraphStyle('CNCap',parent=body,fontSize=8.3,leading=13,alignment=TA_CENTER,spaceAfter=12)
    title_style=ParagraphStyle('CNTitle',parent=head,fontSize=20,leading=29,alignment=TA_CENTER,spaceAfter=18)
    doc=SimpleDocTemplate(str(OUT/'competition_paper.pdf'),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=42,bottomMargin=43,title=title,author='MathorCup project research draft; AI-assisted')
    width=A4[0]-86;story=[Paragraph(html.escape(title),title_style),Paragraph('研究稿 v1 | 真实0–6分校准及部分缺陷阈值仍待补证',caption),Spacer(1,12)]
    for heading,text in sections:
        section_start=len(story)
        story.append(Paragraph(html.escape(heading),head))
        # Images must be block elements even if an author left only one newline.
        text=re.sub(r'(?m)^\s*(!\[.*?\]\(.*?\))\s*$',r'\n\n\1\n\n',text)
        paras=text.split('\n\n')
        for para in paras:
            para=para.strip()
            if not para:continue
            if para.startswith('| '):
                rows=[line.strip().strip('|').split('|') for line in para.splitlines() if line.strip().startswith('|') and not re.match(r'^\|\s*---',line)]
                rows=[[Paragraph(html.escape(c.strip()),small) for c in row] for row in rows]
                n=len(rows[0]);widths=[width/n]*n
                if n==4:widths=[width*.19]+[width*.27]*3
                tab=Table(rows,colWidths=widths,repeatRows=1,hAlign='CENTER')
                tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3eded')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8f8')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#216869'))]))
                story.extend([tab,Spacer(1,10)])
            elif para.startswith('!['):
                match=re.match(r'!\[(.*?)\]\((.*?)\)',para)
                from PIL import Image as PILImage
                p=OUT/match.group(2);w,h=PILImage.open(p).size
                scale=min(width/w,550/h);story.append(KeepTogether([Image(str(p),width=w*scale,height=h*scale),Paragraph(html.escape(match.group(1)),caption)]))
            else:
                for line in para.splitlines():
                    if line.strip():story.append(Paragraph(html.escape(line.strip()),body))
        if heading.startswith('附录B'):
            story[section_start:]=[KeepTogether(story[section_start:])]
    def footer(canvas,doc):
        canvas.setFont('CN',8);canvas.setFillColor(colors.HexColor('#667777'));canvas.drawString(43,25,'MathorCup 啤酒项目 | 研究稿 v1');canvas.drawRightString(A4[0]-43,25,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def checklist():
    rows=[
        ('赛题与真实附件结构核验','完成','task_text、条码审计、原文件哈希与外部来源记录'),
        ('问题1 连续感官预测','完成','50项冻结复现；原测试集不再用于模型选择'),
        ('问题1 原始0–6分校准','未解决','缺原始配对标签/逆变换参数；相对分级不是还原'),
        ('问题1 相对分级与缺失评估','完成','12500条样本外相对等级、界点与遮蔽敏感性'),
        ('问题2 官方矩阵化合物图','完成','218节点、245边；实际Spearman类型核验'),
        ('问题2 全50维Top-3及稳定性','完成','150条候选；200轮bootstrap、风格调整、来源证据'),
        ('问题2 因果或感官实验确认','未完成','统计关联不能替代单体作用实验'),
        ('问题3 风格条件谱生成','完成研究实现','22风格2200谱；保真度、重复、多样性诊断'),
        ('问题3 跨风格迁移','完成研究实现','11款Pils/Lager→Tripel；完整候选记录'),
        ('问题3 缺陷化学字段不增','有限保证','已知字段原生值不变；缺失保持未知，已有缺陷不被消除'),
        ('问题3 trans-2-nonenal及Stale hops阈值','未解决','附件未给出；Stale hops仅模型代理约束'),
        ('物理配方及实测风味合理性','未验证','原生混合尺度统计谱不保证配方恒等式、工艺可行性'),
        ('完整论文、代码与图表','完成研究稿','MD/PDF、独立Notebook、源码、图表和环境清单'),
        ('最终赛事格式与队伍身份','待核对','当前题面无队伍信息及详细格式；不虚构'),
        ('向赛事平台正式提交','未执行','当前工作仅生成本地研究材料'),
    ]
    df=pd.DataFrame(rows,columns=['requirement','status','evidence_or_limitation']);df.to_csv(OUT/'requirements_status.csv',index=False,encoding='utf-8-sig')
    text='# 比赛材料入口与提交状态\n\n本轮已完成三问研究实现和完整论文稿，尚不能声称全部硬要求已满足。\n\n'+table(df)+'\n\n## 使用\n\n- competition_paper.pdf / .md：论文研究稿。\n- q1_*：全部样本外标准化预测与相对分级，**不是原始0–6分还原**。\n- q2_*：官方网络、全50维Top-3与稳定性、化学来源。\n- q3_*：2200条生成谱、血缘、画像、生成验证和11款迁移。\n- reference_provenance.json：原研究公开补充表来源与校验。\n- notebooks/06、07、08：已执行的结果复查笔记本，位于项目notebooks目录。\n\n## 复现\n\n在项目根目录执行：\n\n```powershell\n.venv/Scripts/python.exe -m pip install -r requirements_competition.txt\n.venv/Scripts/python.exe scripts/competition_pipeline.py --stage all\n.venv/Scripts/python.exe scripts/build_competition_deliverables.py\n```\n\n首次结果复查Notebook已存在时，构建脚本会验证并保留它们；不覆盖原Notebook。脚本可重复计算本次独立输出目录的派生结果；如需保留多次实验快照，请先复制目录。旧data、旧outputs顶层文件与旧Notebook不会被写入。\n\n当前生成对象只能由本项目脚本加载，勿加载来源不明的joblib。公开补充表为数据文件，未执行外部脚本。\n'
    (OUT/'START_HERE.md').write_text(text,encoding='utf-8')


def main():
    reference_audit();figures();title,sections=report();pdf(title,sections);checklist()
    if not all((ROOT/'notebooks'/n).exists() for n in ['06_competition_scoring_review.ipynb','07_competition_network_all50.ipynb','08_competition_conditional_generation.ipynb']): notebooks()
    print('Paper and deliverables built:',OUT,flush=True)

if __name__=='__main__':main()
