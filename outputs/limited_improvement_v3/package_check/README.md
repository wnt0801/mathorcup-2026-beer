# MathorCup v4唯一推荐支撑包

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
