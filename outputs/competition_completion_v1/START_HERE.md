# 比赛材料入口与提交状态

本轮已完成三问研究实现和完整论文稿，尚不能声称全部硬要求已满足。

| requirement | status | evidence_or_limitation |
| --- | --- | --- |
| 赛题与真实附件结构核验 | 完成 | task_text、条码审计、原文件哈希与外部来源记录 |
| 问题1 连续感官预测 | 完成 | 50项冻结复现；原测试集不再用于模型选择 |
| 问题1 原始0–6分校准 | 未解决 | 缺原始配对标签/逆变换参数；相对分级不是还原 |
| 问题1 相对分级与缺失评估 | 完成 | 12500条样本外相对等级、界点与遮蔽敏感性 |
| 问题2 官方矩阵化合物图 | 完成 | 218节点、245边；实际Spearman类型核验 |
| 问题2 全50维Top-3及稳定性 | 完成 | 150条候选；200轮bootstrap、风格调整、来源证据 |
| 问题2 因果或感官实验确认 | 未完成 | 统计关联不能替代单体作用实验 |
| 问题3 风格条件谱生成 | 完成研究实现 | 22风格2200谱；保真度、重复、多样性诊断 |
| 问题3 跨风格迁移 | 完成研究实现 | 11款Pils/Lager→Tripel；完整候选记录 |
| 问题3 缺陷化学字段不增 | 有限保证 | 已知字段原生值不变；缺失保持未知，已有缺陷不被消除 |
| 问题3 trans-2-nonenal及Stale hops阈值 | 未解决 | 附件未给出；Stale hops仅模型代理约束 |
| 物理配方及实测风味合理性 | 未验证 | 原生混合尺度统计谱不保证配方恒等式、工艺可行性 |
| 完整论文、代码与图表 | 完成研究稿 | MD/PDF、独立Notebook、源码、图表和环境清单 |
| 最终赛事格式与队伍身份 | 待核对 | 当前题面无队伍信息及详细格式；不虚构 |
| 向赛事平台正式提交 | 未执行 | 当前工作仅生成本地研究材料 |

## 使用

- competition_paper.pdf / .md：论文研究稿。
- q1_*：全部样本外标准化预测与相对分级，**不是原始0–6分还原**。
- q2_*：官方网络、全50维Top-3与稳定性、化学来源。
- q3_*：2200条生成谱、血缘、画像、生成验证和11款迁移。
- reference_provenance.json：原研究公开补充表来源与校验。
- notebooks/06、07、08：已执行的结果复查笔记本，位于项目notebooks目录。

## 复现

在项目根目录执行：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements_competition.txt
.venv/Scripts/python.exe scripts/competition_pipeline.py --stage all
.venv/Scripts/python.exe scripts/build_competition_deliverables.py
```

首次结果复查Notebook已存在时，构建脚本会验证并保留它们；不覆盖原Notebook。脚本可重复计算本次独立输出目录的派生结果；如需保留多次实验快照，请先复制目录。旧data、旧outputs顶层文件与旧Notebook不会被写入。

当前生成对象只能由本项目脚本加载，勿加载来源不明的joblib。公开补充表为数据文件，未执行外部脚本。
