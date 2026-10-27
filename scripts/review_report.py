"""Synchronize v2 claims with audit tables; reuse v1 mathematical implementation."""
import json
import textwrap
import re
import pandas as pd
from review_config import ROOT,OUT


def revised_sections(sections,table):
    top=pd.read_csv(OUT/'q2_top3_all50_evidence.csv')
    cov=pd.read_csv(OUT/'q3_style_diversity_coverage.csv')
    defects=pd.read_csv(OUT/'q3_defect_evidence_summary.csv')
    summary=pd.read_csv(OUT/'q3_summary_consistency_audit.csv')
    guard=json.loads((OUT/'q3_stale_guard_assessment.json').read_text(encoding='utf-8'))
    stable=int(top.candidate_class.eq('stable_association').sum())
    stable_edges=len(pd.read_csv(OUT/'q2_network_edges_stable.csv'))
    rewritten=[]
    for heading,text in sections:
        text=text.replace('outputs/competition_completion_v1','outputs/competition_completion_v2')
        text=text.replace('完整研究稿','审核研究稿')
        text=text.replace('新增非恒定约束解决空约束问题，未解决预测可信度问题','非恒定辅助约束仅提供模型层面的条件，未解决预测可信度问题')
        text=text.replace('既有100轮网络bootstrap以相关强度及同号保留率筛选得到172条稳定边','本轮从原始附件独立重算100轮网络bootstrap，以相关强度及同号保留率筛选得到'+str(stable_edges)+'条稳定边')
        if heading=='摘要':
            text+='\n\n本轮审核在不含v1派生结果的干净副本内重新训练、重建网络、生成、迁移、构建材料及验证；冻结输入边界见附录B与复现说明。辅助Stale hops模型未优于均值基准，因此只报告模型代理条件，不作实际缺陷保证。实际酿造和因果实验未被给定题面列为强制提交条件，本版本将其列为研究局限。'
        elif heading.startswith('3 问题1'):
            text+='\n\n评分口径复核详见Q1_SCORING_SCOPE.md及q1_score_scale_evidence.json；输出字段actual_relative_grade也是连续标签按训练界点分档，不是原始人工标签。现有资料未发现可靠的原始标签/逆变换，未生成原始0–6分列。'
        elif heading.startswith('5 问题2'):
            text+=f'\n\n保留全部150条Top-3，另加candidate_class字段：入选频率≥0.8且风格调整方向一致的{stable}条标为stable_association；频率较低的标为exploratory_association；若排名稳定但方向反转，则单列stable_rank_direction_sensitive。该分类为审阅标签，未据其修改Top-3名单或模型。稳定也不等于因果。'
        elif heading.startswith('8 条件生成'):
            conditional=cov[cov['mode'].eq('conditional_local')].set_index('style')
            resample=cov[cov['mode'].eq('conditional_resample')].set_index('style')
            small=pd.DataFrame({'风格':conditional.index,'训练数':conditional.n_train,'留出数':conditional.n_test,'局部方差比':conditional.generated_train_total_scaled_variance_ratio,'重采样方差比':resample.generated_train_total_scaled_variance_ratio,'局部留出覆盖':conditional.holdout_covered_at_train_nn95}).reset_index(drop=True)
            text+='\n\n### 逐风格多样性和覆盖（新增审核诊断）\n\n方差使用原生观测值、ddof=1，不用缺失填补结果假装方差。总方差先按训练集尺度归一化，再计算生成/同风格训练方差比；恒定/缺失列按CSV中的有效列数说明。距离在同一个训练拟合的填补、标准化空间计算。覆盖半径取同风格训练点留一最近邻距离的95%分位数，留出点到生成集合距离不超过该半径即计覆盖。该经验半径不是置信保证，尤其小风格样本时不能作概率解释。未用这些新增诊断重新选择方法或超参数。\n\n'+table(small)+'\n\n留出样本不足2款的风格，其方差字段保留NA而非0。q3_style_diversity_coverage.csv逐风格逐方法共66行，包含训练/留出数、方差、生成对距离、同风格训练距离、留一距离与覆盖；与条件重采样、无条件基线一并提供。\n\n![图7：逐风格多样性与覆盖](figures/07_diversity_coverage.png)'
            text+=f'\n\n逐风格等权平均下，局部生成/训练方差比为{conditional.generated_train_total_scaled_variance_ratio.mean():.4f}，重采样为{resample.generated_train_total_scaled_variance_ratio.mean():.4f}；留出覆盖分别为{conditional.holdout_covered_at_train_nn95.mean():.4f}和{resample.holdout_covered_at_train_nn95.mean():.4f}。这组诊断表明局部生成有方差收缩，未显示对条件重采样的覆盖优势；不据此反复调参或声称生成已充分覆盖真实分布。'
        elif heading.startswith('9 跨风格'):
            text+='\n\n### 四种缺陷的证据状态\n\n'+table(defects[['field','native_known','native_missing','calibrated_missing','presence_mismatch','numeric_source_conflict','zero_semantics_unknown','threshold_unknown']].rename(columns={'field':'字段','native_known':'原生已知','native_missing':'原生缺失','calibrated_missing':'附件8缺失','presence_mismatch':'缺失状态差异','numeric_source_conflict':'数值冲突','zero_semantics_unknown':'零含义未知','threshold_unknown':'阈值未知'}))
            text+='\n\n这些标记可重叠，不能加总为互斥风险类别。DMS/4VG原生有限值不代表附件8零值含义已确定；双乙酰9处缺失/零冲突与3处数值差异分别统计；trans-2-nonenal还有1处缺失状态差异和2处数值冲突，阈值全部未知。迁移对应的44条证据在q3_transfer_defect_evidence.csv，逐条保留原生已知/缺失、来源差异与阈值标记，不能把缺失冻结视为安全证明。'
            text+=f'\n\n辅助Stale hops五折MAE差（模型减基准）为{guard["paired_mean_MAE_difference"]:.6f}，仅{guard["model_better_folds"]}/5折较优，没有显示平均MAE优势。它没有获得实际安全约束资格；所有满足表述限于“模型代理约束”，不能由优化后预测值降低推出实际陈旧酒花缺陷降低。'
        elif heading.startswith('10 鲁棒性'):
            text+='\n\n### 汇总指标与组分一致性\n\n对题面提及的总酯、酒花香气、总酸、硫化合物和热量五项重新检查附件8元数据、公开补充表方法记录及变换配对。没有找到足以同时确认组分清单、系数、单位换算及变换顺序的完整链条。因此五项均标“未验证”，consistency_error为NA，recomputed=False；不猜sum公式，也不重写生成汇总列。kcalperc虽有单位kcal%，也不能据此推导能量公式。\n\n'+table(summary[['field','unit','status','recomputed']].rename(columns={'field':'汇总字段','unit':'附件单位','status':'组分一致性','recomputed':'已重算'}))
            text+='\n\n统计谱合理性仅由范围、多样性、距离、相关结构等模型诊断支持；感官画像是冻结模型预测；实际酿造有效性本项目未测。分类器一致率与优化目标变化不是感官或酿造实测。酿造和因果实验属于研究局限，而非给定题面额外列出的必须完成提交条件。'
        elif heading.startswith('附录B'):
            text='''本轮推荐入口为python scripts/run_review.py。完整计算从原始附件及frozen_inputs的冻结输入冷启动重算，不读取历史v1训练模型或派生结果。不能把它称为仅从附件即可重做全部历史选模：model_selection_cv_50.csv是冻结的训练CV选择；test_predictions_50.csv只用于历史预测等价性检查；公开补充xlsx只供溯源。角色表、名称映射与网络bootstrap现已从附件重建。详见REPRODUCIBILITY.md。

原始哈希使用历史期望值，不为通过验证而重建；v2路径统一为POSIX相对路径，读取器兼容旧反斜杠键。只有可由同一UTF-8文本的LF/CRLF转换严格解释的变化才标记eol_only，内容修改一律失败。当前工作区35项旧清单均精确匹配；模拟换行和篡改测试分别验证允许和拒绝路径。

06–08旧Notebook保留为v1结果复查；v2在本输出review_notebooks中重新生成并执行的副本仍然是结果复查，不承担完整训练。02和04仅有空白单元格，保留为历史占位，不能作为训练入口。

完整源代码包含训练、生成、审计、验证、干净副本重跑和带时间戳打包。运行日志、环境、输入哈希、验证证据随材料保存。字体可用--font或MATHORCUP_CJK_FONT指定；缺少有效中文TrueType字形时直接给出可操作错误。

AI实际参与了代码、分析解释与材料草拟。当前可提供本轮程序日志和变更，仓库中未发现完整逐轮历史交互导出；不得补造对话或人工工作。若正式规则要求完整AI交互记录，应由队伍提供真实导出并核对声明，当前列为待补/待核对材料。'''
        # Source sections are prose/tables only; remove f-string indentation that
        # would otherwise turn Markdown paragraphs and image links into code blocks.
        text=re.sub(r'(?m)^[ \t]+','',text)
        rewritten.append((heading,text))
    return rewritten


def write_status(table):
    rows=[
        ('问题1 标准化连续预测','required_analysis','已复现','50维；冻结模型不按新测试结果选择'),
        ('问题1 原始0–6分校准','organizer_confirmation','未解决','未找到可靠配对原始标签/参数；相对等级不能替代为真实还原'),
        ('问题1 目标维度口径','organizer_confirmation','待确认','实际A/F30项+其他20项；题面约40项'),
        ('问题2 全50维Top-3、网络与稳定性','required_analysis','已完成研究实现','稳定/探索分层，入选频率、风格调整及化学证据'),
        ('问题3 风格条件统计谱及感官画像','required_analysis','已完成研究实现','2200谱；逐风格样本量、多样性、覆盖、基线对照'),
        ('问题3 原生四缺陷约束','required_analysis','有限证据','已知值完全固定；缺失/零/来源冲突/阈值未知分别报告'),
        ('问题3 实际Stale hops不恶化','organizer_confirmation','证据不足','辅助模型未优于均值，仅可报告模型代理约束；浓度/阈值定义待确认'),
        ('汇总与组成成分恒等式','research_limitation','未验证','缺完整公式/单位/变换证据，五项不计算伪误差、不重算'),
        ('实际酿造和因果实验','research_limitation','未开展','当前题面未列为强制提交条件；不应将缺失实验写作额外硬门槛'),
        ('论文、源码、图表与可复现性','local_quality','本轮重新验证','脚本实际重跑；干净副本记录另见clean_reproduction.json'),
        ('AI参与声明','submission_material','已如实保留','完整历史交互导出未在仓库中发现；按正式规则核对是否需补交'),
        ('队伍信息和正式提交格式','organizer_confirmation','待核对','当前提供PDF未列出详细格式；不虚构编号或匿名规则'),
        ('平台提交及Git推送','not_authorized','未执行','仅本地审核材料，无自动提交、推送'),
    ]
    df=pd.DataFrame(rows,columns=['requirement','category','status','evidence_or_limitation'])
    df.to_csv(OUT/'requirements_status.csv',index=False,encoding='utf-8-sig')
    intro='''# 唯一推荐材料入口：MathorCup 审核版 v2

这是本地可提交审核的研究材料；不能宣称真实0–6评分和实际缺陷保证均已解决。v1文件与ZIP原样保留。

- [论文PDF](competition_paper.pdf) · [可编辑论文](competition_paper.md)
- [第一问评分口径说明](Q1_SCORING_SCOPE.md)
- [本轮验证](verification.json) · [干净副本重跑](clean_reproduction.json)
- [审查总结](../../REVIEW_SUMMARY.md) · [完整复现命令](../../REPRODUCIBILITY.md)
- review_notebooks/06–08是已执行的结果复查；完整训练与审计入口是 scripts/run_review.py。

统计化合物谱、感官模型预测和实际酿造有效性分开解释。分类器一致率及优化后预测不是实测证明。

'''
    (OUT/'START_HERE.md').write_text(textwrap.dedent(intro)+table(df)+'\n',encoding='utf-8')
