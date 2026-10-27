# 来源与有限调查结论

## 原始评分：找到新资料，但编号不能直接连接

原研究：[Schreurs等，2024，Methods / Sensory dataset / Trained panel](https://www.nature.com/articles/s41467-024-46346-0)。方法说明按评分者居中及标准差缩放；并非对250个啤酒均值作单一z变换。评分者出席不齐，标准化后平均通常没有统一的精确逆函数。论文未充分明确零方差与重复场次处理，不能擅自推断。

[Zenodo v4，2026-10-24，DOI 10.5281/zenodo.23624915](https://zenodo.org/records/23624915)新增Raw sensory data anonymized.csv。下载MD5与发布值a8f911d8c0061e096abf3b087678e164一致，SHA256见q1_decision.json。2197条记录、16名匿名评分者、250个匿名啤酒编号，50个评分字段0–6；Beer只有13种值，是场次内标识，不能当项目beer键。beer_name虽然也是1–250，但与附件同数字相连时风格和感官值不一致。

按评分者×维度的样本标准差（ddof=1）居中缩放，再按匿名beer_name求记录均值。以预先指定的A_malt_all、F_malt_all、overall三维精确数值指纹匹配：每款有且仅有一个误差<1e-6的附件行，形成250对一一映射。其余维度独立交叉核实：原200款训练集中38维可复现，最大误差1.64e-8；12维不复现，最大差约0.993。不是按风格或近似最近邻强行配对；数值指纹提供很强的来源对应证据，但不是主办方发布的正式ID映射。明细和第二近匹配间隔见q1_identity_mapping.csv。

未复现的12维：A_hops_woody、A_esters_flower、A_esters_fruity、F_hops_tropical、F_hops_woody、F_esters_flower、F_esters_fruity、stale_hops、clove、lactic、acetic、barnyard。差异集中于存在零方差评分者的项目；尝试将零方差中心化值设0亦未消除差异。原因尚未确认，本轮停止追查，不把这些维度纳入校准。列名A_ester_fruity/F_ester_fruity仅记录为明确单复数别名，两维均不纳入校准。

实际校准目标定义为每款、每维原始评分记录的算术平均。重复场次按记录等权；改为先按评分者求均值，训练集全维平均绝对差约0.00256，但个别值最大0.32038，未隐瞒此口径差异。方法是训练内有监督线性校准，不宣称还原每个评分者的评分或确定官方整数等级。

## 原代码检查

[模型训练代码](https://zenodo.org/records/10653704/files/Machine%20learning%20models%20transcript.py?download=1)第68–86行直接读取S4；Y的StandardScaler行被注释，不能从该脚本获取感官原始逆参数。

[ANOVA代码](https://zenodo.org/records/10653704/files/Sensory%20data%20quality%20control%20-%20ANOVA%20on%20repeated%20samples.Rmd?download=1)第21行引用Raw_Sensory_data_Trained_panel.csv；新发布页明确说新增文件是其匿名版本。该代码处理重复场次ANOVA，不给最终聚合的完整预处理实现。

[主图代码](https://zenodo.org/records/10653704/files/Main%20Figure%20Generator.Rmd?download=1)第269行读取S4；第660–668行的scale_na在后续添加实验区段，不能直接当作250款面板的完整预处理流程。相关网页取回证据保存在sources/retrieval_records.json。未执行下载的外部代码。

## Stale hops与化学尺度

附件8 Sensory names把stale_hops标为Trained panel / Taste / Others / numeric，即陈旧酒花感官描述，并非一列已校准的化学浓度。原论文感官结果指出其与Barnyard常在酸啤酒中共同出现（rho=0.97），这不是可靠化学代理或通用缺陷阈值。没有找到可把该感官指标严格换算成某个化合物浓度的依据。

复用v3字段角色：218个单项测量候选、4个混合组成/聚合性质、5个整体理化指标、4个汇总指标；热量kcalperc属于整体理化角色，但也是题目五项汇总之一，实验单独冻结五项。分类不等于所有218项都有绝对浓度校准。

经确认log10的坐标上，v3原生插值对应浓度的加权几何平均，而不是浓度的线性混合。两者都可保持正值及端点范围，但不能把原生alpha解释为酿造体积比例。浓度候选仅修改每折训练正值配对、单位已知、变换全部匹配的单项化学坐标；未知坐标冻结。其表现不足，未采纳。

DMS、4VG以可信正值配对log10解释，单位分别μg/L、mg/L；双乙酰与反式-2-壬烯醛按一致的原值对解释，单位来自附件8。零值、缺失、数值冲突仍是未知。阈值19 μg/L、0.09 mg/L、0.20 mg/L仅为感知参考，不是经验证的风格缺陷界限；反式-2-壬烯醛缺阈值。缺陷浓度/超出量的逐折报告见q3_defect_concentration_cv.csv。

原始资料和已检查主图/模型代码未提供五项汇总的完整组成、公式和变换链，本轮不猜公式、不重建，维持未验证。
