# 复现、审核与打包

## 输入边界

这是“原附件 + 冻结历史策略”的冷启动重算，不是仅凭原附件重做全部历史选模。

| 输入 | 角色 | 能否从本轮原附件重建 |
| --- | --- | --- |
| data/附件1–8、数据说明、赛题PDF | 受保护原始材料，共10文件 | 原始输入，不改写 |
| frozen_inputs/model_selection_cv_50.csv | 已冻结的50维训练CV选择 | 本轮不重新挑模型；必须随材料提供，不能从测试成绩重建 |
| frozen_inputs/test_predictions_50.csv | 历史预测比较基准 | 只校验冻结复现，不是训练标签；不能称为新盲测 |
| frozen_inputs/reference_supplement.xlsx | 已缓存公开补充表，用于评分/变换来源审计 | 外部来源快照，不依赖联网下载；MD5及SHA保留 |
| 特征角色、化学名称映射、网络及网络bootstrap | 派生结果 | v2从原始附件重建，不再读取旧outputs对应CSV |
| 模型、连续预测、相对等级、生成谱、迁移与诊断 | 派生结果 | v2脚本重算，不载入v1模型或派生结果 |

`frozen_inputs/source_sha256.json`保留13个实际运行输入的历史期望摘要。data及两个冻结CSV的期望值来自v1原清单，补充表摘要来自v1支撑文件清单；没有通过重新计算当前文件摘要来掩盖变化。`legacy_original_sha256.json`仅将旧35项清单键改为相对正斜杠路径，摘要值不变，供历史追踪；它并不意味着旧35项派生文件都是v2运行依赖。

## 哈希策略

读取器兼容 `data\1.csv` 和 `data/1.csv`。优先要求原始字节SHA256精确匹配。只有已知文本类型、能解码为UTF-8、不含二进制NUL/孤立CR，且LF或CRLF整体转换后的摘要恰好等于历史期望值时，才接受并标记 `eol_only`。不去空格、不改字符、不重新序列化JSON、不改变预期摘要；真正内容修改和二进制变动一律失败。原文件不被规范化写回。

本轮开始旧35项均精确匹配，没有证据表明当前文件发生了Git换行转换。针对性测试另模拟LF/CRLF转换和真实内容篡改，验证前者可解释、后者被拒绝。这不等同已经在Linux/macOS执行过完整流程。

## 环境与字体

推荐使用已记录的Python 3.13；实际版本与依赖见输出的 `dependencies_actual.json`、`environment.json`。requirements_competition.txt锁定当前验证版本；原requirements.txt保留。

Windows PowerShell（在项目根目录）：

```powershell
py -3.13 -m venv .venv_review
& ./.venv_review/Scripts/python.exe -m pip install -r requirements_competition.txt
& ./.venv_review/Scripts/python.exe -m pip check
& ./.venv_review/Scripts/python.exe scripts/run_review.py --output outputs/competition_completion_v2
& ./.venv_review/Scripts/python.exe scripts/verify_competition.py
& ./.venv_review/Scripts/python.exe scripts/package_competition.py
```

若推荐输出目录已经包含完成模型，统一入口会停止以保留该轮结果。另选目录：

```powershell
$env:MATHORCUP_OUTPUT = 'C:/path/to/project/outputs/my_review_run'
& ./.venv_review/Scripts/python.exe scripts/run_review.py --output $env:MATHORCUP_OUTPUT --font 'C:/path/to/CJK-font.ttf'
& ./.venv_review/Scripts/python.exe scripts/verify_competition.py
& ./.venv_review/Scripts/python.exe scripts/package_competition.py
```

macOS/Linux命令（提供运行方式，**未声称本轮在这些系统实跑**）：

```bash
python3.13 -m venv .venv_review
.venv_review/bin/python -m pip install -r requirements_competition.txt
export MATHORCUP_OUTPUT="$PWD/outputs/my_review_run"
.venv_review/bin/python scripts/run_review.py --output "$MATHORCUP_OUTPUT" --font /path/to/CJK-font.ttf
.venv_review/bin/python scripts/verify_competition.py
.venv_review/bin/python scripts/package_competition.py
```

字体支持 `--font` 或 `MATHORCUP_CJK_FONT`；也会查找项目fonts目录、常见Linux/macOS中文TrueType字体及Windows字体。选定前会检查中文字符覆盖和ReportLab可用性，找不到时给出明确错误，绝不把只有西文字形的字体静默用于中文PDF。部分CFF格式OTF不受ReportLab支持，应选择可用的TrueType字体/集合。字体许可由提供方决定，本项目不擅自打包系统字体。

## 实际运行与验证

`scripts/run_review.py`依次执行：pip依赖检查 → 8项针对性回归测试 → 全50维冻结训练及OOF → 关联/网络bootstrap → 条件生成及迁移 → 来源/缺陷/汇总/覆盖审计 → 论文、图、结果复查Notebook → 独立验证。每阶段返回码、耗时、UTC时间与日志都会保存；出错立即停止。

`scripts/clean_copy_reproduce.py`新建时间戳目录，只复制data、frozen_inputs、脚本、依赖及说明文件，不复制旧outputs或训练模型，用当前Python环境完整运行上述流程。该实验验证了干净项目副本，不等于从空白虚拟环境重新联网安装依赖；实际证据见clean_reproduction.json。运行后保留该副本，不递归删除任何历史材料。

针对性验证包括：训练/测试及OOF界点范围、历史预测复现、模型序列化、生成血缘与缺失传播、66条风格/方法方差及覆盖重算、四缺陷已知/缺失/冲突、辅助代理CV、原生值约束、图的源数据摘要与论文口径。验证文件含本轮时间戳，旧passed不作为本轮证据。

## 论文构建和重复打包

需要单独重建论文时，在对应输出环境下执行：

```powershell
& ./.venv_review/Scripts/python.exe scripts/build_competition_deliverables.py --font 'C:/path/to/CJK-font.ttf'
& ./.venv_review/Scripts/python.exe scripts/verify_competition.py
```

每次打包均生成包含UTC时间和唯一后缀的 `mathorcup_support_v2_*.zip`，不覆盖旧ZIP。ZIP中保存独立清单，每个文件按包内相对正斜杠路径记录SHA256；创建后检查CRC并逐项重读摘要。打包程序允许连续运行；包只包含当前审核版本的必要代码、原始/冻结输入、结果和说明，不把所有历史输出塞进运行依赖。

Python脚本的默认输出为v2。若设置了MATHORCUP_OUTPUT，运行、单独验证和打包须使用相同值。06–08只复查结果；02和04是历史空白占位。AI使用声明保留真实参与事实，完整历史交互记录缺失不得补造，是否提交应按正式规则确认。
