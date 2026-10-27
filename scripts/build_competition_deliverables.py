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
import argparse
import os

import numpy as np
import pandas as pd

from review_config import ROOT, OUT, FROZEN, chinese_font

def table(df,columns=None):
    if columns is not None:df=df[columns]
    names=list(df.columns)
    rows=[]
    for row in df.itertuples(index=False,name=None):
        rows.append(['NA' if pd.isna(v) else f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row])
    return '| '+' | '.join(map(str,names))+' |\n| '+' | '.join(['---']*len(names))+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)


def reference_audit():
    p=FROZEN/'reference_supplement.xlsx'
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
    font=chinese_font()
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'savefig.dpi':220,'axes.spines.top':False,'axes.spines.right':False})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    def export(fig,name):
        fig.savefig(figdir/(name+'.png'),bbox_inches='tight');fig.savefig(figdir/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    m=pd.read_csv(OUT/'q1_metrics.csv').query("split=='heldout_test'").sort_values('target')
    fig,ax=plt.subplots(figsize=(9,12));pos=np.arange(len(m));ax.barh(pos-.18,m.baseline_MAE,height=.35,label='训练均值基准',color='#b8c2ca');ax.barh(pos+.18,m.MAE,height=.35,label='冻结模型',color='#216869');ax.set_yticks(pos,m.target,fontsize=8);ax.invert_yaxis();ax.set_xlabel('测试MAE（附件4标准化尺度）');ax.legend();ax.set_title('50项感官预测：测试结果复现');fig.tight_layout();export(fig,'01_prediction_errors')
    sensitivity=pd.read_csv(OUT/'q1_missingness_by_group.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    for group,label,color in [('Aroma','香气15维','#216869'),('Flavour','风味15维','#c39243'),('All50','全部50维','#815c98')]:
        d=sensitivity.loc[sensitivity['group'].eq(group)].groupby('additional_missing_rate').MAE.agg(['mean','std'])
        ax.errorbar(d.index*100,d['mean'],yerr=d['std'],marker='o',capsize=4,color=color,label=label)
    ax.set_xlabel('历史测试输入额外遮蔽比例（%）');ax.set_ylabel('标准化目标平均MAE');ax.set_title('固定模型的缺失敏感性（每档5次）');ax.legend();fig.tight_layout();export(fig,'02_missingness')
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
    prof=pd.read_csv(OUT/'q3_style_sensory_profiles.csv');targets=['A_malt_all','A_hops_all','A_esters_all','F_malt_all','F_hops_all','F_esters_all','acidity','bitternes','sweetness','body','alcohol','overall'];mat=prof.pivot(index='style',columns='target',values='predicted_mean').loc[['Tripel','Kriek','Lambic'],targets];fig,ax=plt.subplots(figsize=(10,3.4));implot=ax.imshow(mat,cmap='RdBu_r',vmin=-1.5,vmax=1.5,aspect='auto');ax.set_xticks(np.arange(len(targets)),['麦芽香气','酒花香气','酯类香气','麦芽风味','酒花风味','酯类风味','酸度','苦味','甜度','醇厚度','酒精感','总体喜好'],rotation=35,ha='right',fontsize=8);ax.set_yticks(np.arange(3),mat.index)
    for i in range(3):
        for j in range(len(targets)):ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(implot,ax=ax,label='预测标准化得分');ax.set_title('三种风格的生成感官画像（非原始0–6分）');fig.tight_layout();export(fig,'05_style_profiles')
    tr=pd.read_csv(OUT/'q3_transfer_summary.csv');fig,ax=plt.subplots(figsize=(8,4));p=np.arange(len(tr));ax.bar(p-.18,tr.source_target_distance,width=.36,label='修改前',color='#b8c2ca');ax.bar(p+.18,tr.transferred_target_distance,width=.36,label='修改后',color='#216869');ax.set_xticks(p,tr.source_beer);ax.set_xlabel('Pils/Lager beer编号');ax.set_ylabel('到Tripel训练谱的最小RMS距离');ax.set_title('被优化的目标距离（不是独立迁移验证）');ax.legend();fig.tight_layout();export(fig,'06_transfer')
    cov=pd.read_csv(OUT/'q3_style_diversity_coverage.csv')
    styles=['Tripel','Kriek','Lambic'];pos=np.arange(len(styles));fig,(a,b)=plt.subplots(2,1,figsize=(7,6),sharex=True)
    for mode,label,color,offset in [('conditional_local','条件局部','#216869',-.18),('conditional_resample','条件重采样','#c39243',.18)]:
        d=cov.loc[cov['mode'].eq(mode)].set_index('style').loc[styles]
        a.bar(pos+offset,d.generated_train_total_scaled_variance_ratio,width=.36,label=label,color=color)
        b.plot(pos,d.holdout_covered_at_train_nn95,marker='o',label=label,color=color)
    a.axhline(1,color='gray',linewidth=.8);a.set_ylabel('生成/训练方差比');a.set_title('主要风格的多样性及经验覆盖');a.legend(fontsize=9)
    counts=cov.loc[cov['mode'].eq('conditional_local')].set_index('style')
    b.set_ylabel('留出覆盖（训练NN95半径）');b.set_ylim(-.05,1.05);b.set_xticks(pos,[s+'\n训练/留出='+str(int(counts.at[s,'n_train']))+'/'+str(int(counts.at[s,'n_test'])) for s in styles],fontsize=9);b.legend(fontsize=9);fig.tight_layout();export(fig,'07_diversity_coverage')
    compounds=pd.read_csv(OUT/'q3_transfer_compound_changes.csv');sensory=pd.read_csv(OUT/'q3_transfer_sensory_summary.csv').set_index('target')
    selected=['ethanol..v.v.','ethyl_acetate','ethyl_hexanoate','isoamyl_alcohol','bitterness','pH'];chem=compounds.groupby('field').delta_train_SD.mean().reindex(selected)
    targets=['F_malt_all','F_hops_all','F_esters_all','alcohol','body','bitternes'];change=sensory.loc[targets,'paired_mean_delta']
    fig,(a,b)=plt.subplots(1,2,figsize=(10,4));a.barh(np.arange(6),chem,color='#216869');a.set_yticks(np.arange(6),['乙醇','乙酸乙酯','己酸乙酯','异戊醇','苦味理化指标','pH']);a.axvline(0,color='gray',linewidth=.8);a.set_xlabel('平均原生变化 / 训练SD');a.set_title('化合物及理化坐标变化')
    b.barh(np.arange(6),change,color='#c39243');b.set_yticks(np.arange(6),['麦芽整体风味','酒花整体风味','酯类整体风味','酒精感','醇厚度','苦味']);b.axvline(0,color='gray',linewidth=.8);b.set_xlabel('平均预测标准化分差');b.set_title('主要感官模型预测变化');fig.tight_layout();export(fig,'08_transfer_changes')
    inputs=['q1_metrics.csv','q1_missingness_by_group.csv','q2_official_compound_graph.graphml','split_manifest.csv','q3_generated_native_spectra.csv','q3_generation_lineage.csv','q3_style_sensory_profiles.csv','q3_transfer_summary.csv','q3_style_diversity_coverage.csv','q3_transfer_compound_changes.csv','q3_transfer_sensory_summary.csv']
    inputs += [p.relative_to(OUT).as_posix() for p in figdir.glob('*') if p.suffix in ['.png','.svg']]
    (figdir/'figure_sources.json').write_text(json.dumps({k:hashlib.sha256((OUT/k).read_bytes()).hexdigest() for k in inputs},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'font_evidence.json').write_text(json.dumps({'selected_font':str(font),'sha256':hashlib.sha256(font.read_bytes()).hexdigest(),'glyph_validation':'啤酒评分模型约束审查','platform_limit':'tested on current host only; configurable lookup is not a Linux/macOS execution claim'},ensure_ascii=False,indent=2),encoding='utf-8')


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
        folder=OUT/'review_notebooks';folder.mkdir(exist_ok=True)
        p=folder/name
        if p.exists():
            old=nbf.read(p,as_version=4);nbf.validate(old)
            assert all(c.execution_count is not None for c in old.cells if c.cell_type=='code')
            print('Preserved existing review notebook:',p.name,flush=True)
            continue
        nb=nbf.v4.new_notebook();nb.cells=[nbf.v4.new_markdown_cell('# '+title+'\n\n这是结果复查Notebook，不是完整训练流程。完整计算入口为 scripts/run_review.py。'),nbf.v4.new_code_cell("from pathlib import Path\nimport os\nimport pandas as pd\nfrom IPython.display import display, Image\nOUT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'q1_metrics.csv').is_file()), None)\nif OUT is None:\n    ROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'scripts/run_review.py').is_file()), None)\n    if ROOT is None: raise FileNotFoundError('请从项目目录或输出目录启动Notebook')\n    OUT=Path(os.environ.get('MATHORCUP_OUTPUT',str(ROOT/'outputs/competition_candidate_v3')))\npd.set_option('display.max_rows',160)\npd.set_option('display.max_columns',25)")]
        nb.cells[1].source=nb.cells[1].source.replace("str(ROOT/'outputs/competition_candidate_v3')","str(ROOT/'results' if (ROOT/'results').is_dir() else ROOT/'outputs/competition_candidate_v3')")
        if name.startswith('08_'):
            blocks.append(('code',"display(pd.read_csv(OUT/'q3_style_diversity_coverage.csv')); display(pd.read_csv(OUT/'q3_summary_consistency_audit.csv')); display(pd.read_csv(OUT/'q3_defect_evidence_summary.csv')); import json; print(json.loads((OUT/'q3_stale_guard_assessment.json').read_text(encoding='utf-8')))"))
            blocks.append(('code',"display(pd.read_csv(OUT/'q3_transfer_compound_summary.csv')); display(pd.read_csv(OUT/'q3_transfer_sensory_summary.csv')); display(pd.read_csv(OUT/'q3_transfer_summary.csv')[['source_beer','before_diagnostic_style','after_diagnostic_style']])"))
        if name.startswith('06_'):
            blocks.append(('code',"display(pd.read_csv(OUT/'q1_group_metrics.csv')); comparison=pd.read_csv(OUT/'q1_dimension_comparison.csv');display(comparison.loc[comparison.comparison.ne('better')])"))
        for kind,text in blocks:
            if kind=='复现':text='完整流程：python scripts/run_review.py。这里仅读取结果；需要13项受校验输入，其中两项是冻结历史策略/比较文件，详见REPRODUCIBILITY.md。'
            nb.cells.append(nbf.v4.new_code_cell(text) if kind=='code' else nbf.v4.new_markdown_cell('## '+kind+'\n\n'+text))
        nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
        from jupyter_client import KernelManager
        (OUT/'.runtime/ipython').mkdir(parents=True,exist_ok=True)
        (OUT/'.runtime/jupyter').mkdir(parents=True,exist_ok=True)
        os.environ['IPYTHONDIR']=str(OUT/'.runtime/ipython')
        os.environ['JUPYTER_RUNTIME_DIR']=str(OUT/'.runtime/jupyter')
        km=KernelManager(kernel_name='python3');km.kernel_spec.argv[0]=sys.executable
        try:
            NotebookClient(nb,km=km,timeout=120,resources={'metadata':{'path':str(OUT)}}).execute()
        finally:
            if km.has_kernel:km.shutdown_kernel(now=True)
            km.cleanup_resources()
        nbf.write(nb,p);print('executed review notebook',name,flush=True)


def report():
    from review_report import paper
    return paper(table)


def pdf(title,sections):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4
    font=chinese_font()
    pdfmetrics.registerFont(TTFont('CN',str(font),subfontIndex=0))
    glyphs=pdfmetrics.getFont('CN').face.charToGlyph
    missing={c for c in title+''.join(h+b for h,b in sections) if not c.isspace() and ord(c) not in glyphs}
    if missing:raise ValueError('Paper contains unsupported font glyphs: '+repr(sorted(missing)))
    styles=getSampleStyleSheet()
    body=ParagraphStyle('CNBody',fontName='CN',fontSize=9.2,leading=15.5,spaceAfter=7,wordWrap='CJK')
    head=ParagraphStyle('CNHead',parent=body,fontSize=14,leading=20,spaceBefore=14,spaceAfter=9,keepWithNext=True,textColor=colors.HexColor('#164748'))
    subhead=ParagraphStyle('CNSubHead',parent=head,fontSize=11,leading=16,spaceBefore=8,spaceAfter=6)
    small=ParagraphStyle('CNSmall',parent=body,fontSize=7.2,leading=11,spaceAfter=0)
    caption=ParagraphStyle('CNCap',parent=body,fontSize=8.3,leading=13,alignment=TA_CENTER,spaceAfter=12)
    title_style=ParagraphStyle('CNTitle',parent=head,fontSize=20,leading=29,alignment=TA_CENTER,spaceAfter=18)
    doc=SimpleDocTemplate(str(OUT/'competition_paper.pdf'),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=42,bottomMargin=43,title=title,author='MathorCup project research draft; AI-assisted')
    width=A4[0]-86;story=[Paragraph(html.escape(title),title_style),Paragraph('候选提交版 v3',caption),Spacer(1,12)]
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
                    if line.strip():
                        stripped=line.strip()
                        story.append(Paragraph(html.escape(stripped[4:]),subhead) if stripped.startswith('### ') else Paragraph(html.escape(stripped),body))
        if heading.startswith('附录B'):
            story[section_start:]=[KeepTogether(story[section_start:])]
    def footer(canvas,doc):
        canvas.setFont('CN',8);canvas.setFillColor(colors.HexColor('#667777'));canvas.drawString(43,25,'MathorCup 啤酒项目 | 候选提交版 v3');canvas.drawRightString(A4[0]-43,25,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def checklist():
    from review_report import write_status
    write_status(table)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--font');parser.add_argument('--output');args=parser.parse_args()
    if args.font:os.environ['MATHORCUP_CJK_FONT']=args.font
    reference_audit();figures();title,sections=report();pdf(title,sections);checklist();notebooks()
    print('Paper and deliverables built:',OUT,flush=True)

if __name__=='__main__':main()
