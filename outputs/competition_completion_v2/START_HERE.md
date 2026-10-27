# 唯一推荐材料入口：MathorCup 审核版 v2

这是本地可提交审核的研究材料；不能宣称真实0–6评分和实际缺陷保证均已解决。v1文件与ZIP原样保留。

- [论文PDF](competition_paper.pdf) · [可编辑论文](competition_paper.md)
- [第一问评分口径说明](Q1_SCORING_SCOPE.md)
- [本轮验证](verification.json) · [干净副本重跑](clean_reproduction.json)
- [审查总结](../../REVIEW_SUMMARY.md) · [完整复现命令](../../REPRODUCIBILITY.md)
- review_notebooks/06–08是已执行的结果复查；完整训练与审计入口是 scripts/run_review.py。

统计化合物谱、感官模型预测和实际酿造有效性分开解释。分类器一致率及优化后预测不是实测证明。

| requirement | category | status | evidence_or_limitation |
| --- | --- | --- | --- |
| 问题1 标准化连续预测 | required_analysis | 已复现 | 50维；冻结模型不按新测试结果选择 |
| 问题1 原始0–6分校准 | organizer_confirmation | 未解决 | 未找到可靠配对原始标签/参数；相对等级不能替代为真实还原 |
| 问题1 目标维度口径 | organizer_confirmation | 待确认 | 实际A/F30项+其他20项；题面约40项 |
| 问题2 全50维Top-3、网络与稳定性 | required_analysis | 已完成研究实现 | 稳定/探索分层，入选频率、风格调整及化学证据 |
| 问题3 风格条件统计谱及感官画像 | required_analysis | 已完成研究实现 | 2200谱；逐风格样本量、多样性、覆盖、基线对照 |
| 问题3 原生四缺陷约束 | required_analysis | 有限证据 | 已知值完全固定；缺失/零/来源冲突/阈值未知分别报告 |
| 问题3 实际Stale hops不恶化 | organizer_confirmation | 证据不足 | 辅助模型未优于均值，仅可报告模型代理约束；浓度/阈值定义待确认 |
| 汇总与组成成分恒等式 | research_limitation | 未验证 | 缺完整公式/单位/变换证据，五项不计算伪误差、不重算 |
| 实际酿造和因果实验 | research_limitation | 未开展 | 当前题面未列为强制提交条件；不应将缺失实验写作额外硬门槛 |
| 论文、源码、图表与可复现性 | local_quality | 本轮重新验证 | 脚本实际重跑；干净副本记录另见clean_reproduction.json |
| AI参与声明 | submission_material | 已如实保留 | 完整历史交互导出未在仓库中发现；按正式规则核对是否需补交 |
| 队伍信息和正式提交格式 | organizer_confirmation | 待核对 | 当前提供PDF未列出详细格式；不虚构编号或匿名规则 |
| 平台提交及Git推送 | not_authorized | 未执行 | 仅本地审核材料，无自动提交、推送 |
