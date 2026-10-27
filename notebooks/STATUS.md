# Notebook与脚本状态

| 文件 | 状态 | 作用 |
| --- | --- | --- |
| 01_data_check.ipynb | 历史交互分析，保留原样 | 包含早期试验及输出；并非当前统一执行入口 |
| 02_prediction.ipynb | 历史空白占位 | 1个空白单元格；不作为有效训练入口 |
| 03_compound_analysis.ipynb | 历史关联分析 | 保留原结果和来源，不覆盖 |
| 04_style_generation.ipynb | 历史空白占位 | 1个空白单元格；不作为有效生成入口 |
| 05_defect_concentration_audit.ipynb | 历史独立核验 | 四化合物浓度、源格和来源差异检查，保留 |
| 根目录06、07、08 | v1结果复查 | 读取v1输出；不是完整训练流程，保留原执行结果 |
| outputs/competition_completion_v2/review_notebooks/06–08 | v2结果复查 | 由本轮干净流程生成并执行；主要展示审计表与图，仍不是完整训练流程 |

唯一完整计算入口：`python scripts/run_review.py`。它依次运行依赖检查、针对性测试、冻结训练与生成、来源审计、论文构建、Notebook结果复查和独立验证。输入依赖及冻结边界见项目REPRODUCIBILITY.md。
