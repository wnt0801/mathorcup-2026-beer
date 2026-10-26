# MathorCup 啤酒项目：三问研究实现

当前材料入口：[START_HERE](outputs/competition_completion_v1/START_HERE.md)

- [完整论文PDF研究稿](outputs/competition_completion_v1/competition_paper.pdf)
- [可编辑Markdown论文](outputs/competition_completion_v1/competition_paper.md)
- [逐项要求与未决问题](outputs/competition_completion_v1/requirements_status.csv)
- [独立验证记录](outputs/competition_completion_v1/verification.json)

新增Notebook：

- [06 冻结预测、缺失评估与相对分级](notebooks/06_competition_scoring_review.ipynb)
- [07 全50维Top-3与化合物网络](notebooks/07_competition_network_all50.ipynb)
- [08 条件生成与缺陷约束迁移](notebooks/08_competition_conditional_generation.ipynb)

原始文件及旧输出保留。新结果单独存放于outputs/competition_completion_v1。

注意：真实原始0–6评分的逆变换仍缺少证据，相对分位数等级不是还原；反式-2-壬烯醛与Stale hops阈值尚未解决。生成与迁移属于统计研究实现，不是已完成酿造/盲评验证的配方，不能宣称所有比赛硬要求已满足。

## 复现

使用Python 3.13与requirements_competition.txt中的实际环境版本。在项目根目录执行：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements_competition.txt
.venv/Scripts/python.exe scripts/competition_pipeline.py --stage all
.venv/Scripts/python.exe scripts/build_competition_deliverables.py
.venv/Scripts/python.exe scripts/verify_competition.py
```

生成模型对象前后使用固定随机种子42；模型按历史训练集选择冻结，本轮没有用测试集成绩继续选模。公开补充表用于溯源，文件随支撑材料附带，MD5已验证。

该流程可重算本次独立目录中的派生文件，不写入旧outputs顶层文件、data文件或旧Notebook。若要保留本轮目录的多次运行快照，先复制该目录。已有新增复查Notebook不覆盖，完整计算请以脚本为准。
