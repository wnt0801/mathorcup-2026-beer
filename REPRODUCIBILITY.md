# v3复现与打包

当前成果目录为outputs/competition_candidate_v3，入口见[START_HERE](outputs/competition_candidate_v3/START_HERE.md)。论文、计算、审计、验证和支撑包均使用此目录；v1/v2不参与本轮结果验证。

## 输入与环境

需要data内10个原始文件，以及frozen_inputs内模型选择CSV、历史测试预测CSV、公开补充工作簿，共13项受SHA256保护的运行输入。期望摘要保留于frozen_inputs/source_sha256.json，不重新采集期望值掩盖变化。仅严格可解释的UTF-8文本LF/CRLF整体转换可标记为eol_only；内容变化失败。原文件不写回。

这是“原附件＋冻结历史选择”的重算，不是从附件重新完成历史选模。历史测试预测只用于比较，不能作为训练标签或新盲测。特征角色、名称映射、网络及全部模型/预测由当前脚本重建。

本轮使用现有.venv_candidate，Python3.13，全部16项声明版本匹配requirements_competition.txt，pip check通过；没有宣称本轮从空白环境联网安装。实际版本见成果目录fresh_environment.json及dependencies_actual.json。

## 完整运行

从项目根目录运行。已有完成模型的目录受到保护，重算请使用新目录：

```powershell
py -3.13 -m venv .venv_candidate
& ./.venv_candidate/Scripts/python.exe -m pip install -r requirements_competition.txt
& ./.venv_candidate/Scripts/python.exe -m pip check
& ./.venv_candidate/Scripts/python.exe scripts/run_review.py --output outputs/my_v3_reproduction
```

已有环境时跳过创建和安装。本轮实际命令是scripts/run_review.py --output outputs/competition_candidate_v3 --no-package，先完成视觉检查和说明同步，再单独打包。

统一入口通过MATHORCUP_OUTPUT把目录传给各子进程；所有独立阶段也接受--output，其优先级高于环境变量。完整运行依次执行依赖检查、11项契约测试、三问及审计、论文及3本复查Notebook构建、独立验证。每阶段返回码和耗时保留在run_summary.json；UTC起止、完整命令和日志在.review/run_execution.json及相邻日志内。失败停止，不用旧passed替代。

独立操作已完成的v3结果：

```powershell
& ./.venv_candidate/Scripts/python.exe scripts/build_competition_deliverables.py --output outputs/competition_candidate_v3
& ./.venv_candidate/Scripts/python.exe scripts/verify_competition.py --output outputs/competition_candidate_v3
& ./.venv_candidate/Scripts/python.exe scripts/package_competition.py --output outputs/competition_candidate_v3
```

只有修改相关材料后才重建/复核，不应重复运行完整训练覆盖已完成结果。打包前先同步根目录REVIEW_SUMMARY.md。已有同名支撑包会被保护而报错，不自动覆盖。

字体支持--font或MATHORCUP_CJK_FONT，自动查找包含中文字形、ReportLab可用的TrueType字体；没有可用字体会明确失败。系统字体不随包分发。PDF用Poppler逐页渲染并人工视觉检查，记录在pdf_visual_review.json。

macOS/Linux提供运行方式，但未在本轮实跑：

```bash
python3.13 -m venv .venv_candidate
.venv_candidate/bin/python -m pip install -r requirements_competition.txt
.venv_candidate/bin/python scripts/run_review.py --output outputs/my_v3_reproduction --font /path/to/CJK.ttf
```

## 包内复现

唯一推荐归档为outputs/competition_candidate_v3/submission_candidate_v3.zip。归档只含本轮必要代码、data、frozen_inputs及results，不混入历史版本、缓存和旧验证。结果路径统一为results，README相对链接逐项检查。SUPPORT_MANIFEST.json记录SHA256；创建后检查CRC、解压文件摘要，并从解压目录独立运行验证。结果见package_verification.json。

解压后可执行python scripts/verify_competition.py --output results；完整重算执行python scripts/run_review.py，默认创建新时间戳目录，保留results。包内可移植JSON明确调整了运行位置字段，数值结果未修改。

原始评分校准、缺陷阈值/来源冲突及五项汇总公式仍未解决。实际酿造和因果实验是研究局限，不是题面新增的强制条件。完整历史AI交互导出未发现，按正式规则向主办方确认，不补造。
