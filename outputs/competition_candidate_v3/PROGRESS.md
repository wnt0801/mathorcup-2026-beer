# v3收尾完成

2026-10-27完成本轮可独立执行的工作。唯一推荐入口：[START_HERE](START_HERE.md)；[最终论文PDF](competition_paper.pdf)、[可编辑稿](competition_paper.md)、[唯一推荐支撑包](submission_candidate_v3.zip)。旧v1/v2及原始/冻结输入保留，不引用旧验证作为v3证据。

## 实际完成与验证

- .venv_candidate全部16项声明依赖版本匹配，pip check通过，11项契约测试通过。
- UTC 09:39:00至09:44:18完整运行三问、来源审计、论文和3本Notebook、独立验证，各阶段返回0；详见run_summary.json和.review/run_execution.json。
- 生成12500条预测、150条Top-3、2200条统计谱和11款迁移；新增分组评估、300条预处理证据、网络解释、2541条化合物变化和550条感官变化已实际运行。
- 发现PDF上标字形缺失后改为等价^2，重建成功；最终8页全部经Poppler渲染及逐页检查，见pdf_visual_review.json。新增的感官预测重放、变化均值、训练SD尺度和网络边数独立核验通过，最终验证返回0，见final_validation_execution.json。
- UTC 09:52:28至09:53:17打包，用时49.19秒，返回0；126个负载文件、22处Markdown链接、CRC、逐文件SHA256、解压完整性及解压目录独立验证全部通过。见package_execution.json、package_verification.json。
- 200个受保护原附件、冻结输入、v1/v2和旧ZIP逐字节未变，见history_preservation.json。首页、兼容入口、复现说明、审查总结及Notebook状态已指向v3。

## 未解决事项

原始0–6评分不能由现有资料唯一还原；相对七级不可冒称原始评分。Stale hops仍未优于均值基准，只是弱模型代理约束。四化合物的未知、单位、变换和来源冲突均保留，实际缺陷不恶化尚不能保证。五项汇总公式仍未验证，不猜公式重算。目标距离下降及模型分类不是独立的真实迁移证明。

请主办方确认评分维度/校准/评估口径、缺陷阈值及来源冲突、五项汇总定义、正式格式和AI记录要求。实际酿造和因果实验是研究局限，不是本题新增强制条件。本轮没有环境阻碍导致的未完成产物；上述题意/数据缺口未伪装成已解决。未执行git commit、git push或赛事平台提交。
