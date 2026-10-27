# MathorCup 啤酒项目

唯一推荐入口：[v3论文与支撑材料](outputs/competition_candidate_v3/START_HERE.md)。

- [论文PDF](outputs/competition_candidate_v3/competition_paper.pdf) · [可编辑Markdown](outputs/competition_candidate_v3/competition_paper.md)
- [唯一推荐支撑包](outputs/competition_candidate_v3/submission_candidate_v3.zip)
- [复现说明](REPRODUCIBILITY.md) · [审查总结](REVIEW_SUMMARY.md) · [v3进度](outputs/competition_candidate_v3/PROGRESS.md)

沿用冻结训练CV选择，完成三问计算、Aroma/Flavour分组评估、预处理核验、网络解释及迁移前后变化分析。全50维历史测试MAE为0.3380；测试集曾被查看，本轮是冻结复现，不是新盲测。

相对0–6级是训练分位数等级，不是原始评分还原。Stale hops只作为弱模型代理约束，不能保证实际缺陷不恶化；四项缺陷保留已知/未知、单位、变换及来源冲突。五项汇总公式仍未验证。迁移目标距离下降不等于独立证明真实风格转换。

原附件、冻结输入、历史Notebook及v1/v2成果保留；旧版验证不作为v3证据。完整训练入口为scripts/run_review.py，复查Notebook只展示结果。本轮不执行git commit、git push或赛事平台提交。AI参与如实说明，缺失的历史交互记录不补造。
