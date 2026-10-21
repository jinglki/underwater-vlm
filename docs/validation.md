# 本次验证记录

验证日期：2026年10月21日。

## 已执行

- 自动测试共 41 项，使用 Python unittest，不依赖 9B 模型权重。
- 默认任务明确为 Qwen3.5 9B 上的指定目标存在性识别，默认批次为 8。
- presence 与 classification 的 JSON 协议、类型、数组长度、顺序及标签限制测试。
- 存在性真值兼容旧 count 标签，并拒绝相互冲突的标签。
- manifest 相对路径、prepared 显式前缀映射、多条件隔离、缺失文件与重复 ID 检查。
- 批量结果顺序、批次尾部、OOM 拆分和后续批次调整。
- 无效回答保留原文并记录失败，不强制解释为目标不存在。
- 小规模运行后恢复、人工中断后恢复、完整断点不加载模型、输入变化拒绝恢复。
- 不完整末行恢复与完整损坏记录拒绝、跨进程运行锁的本机分支。
- 新旧 Processor 参数分发、左 padding、提示词裁剪、多图请求隔离和图片资源释放。
- 完整的 prepare → infer 16 条 → resume → evaluate 流程，生成 25 个条件。
- 该流程的经典增强使用实际实现；测试时仅将 Zero-DCE 与 WaterNet 替换为确定性模拟函数，模型也为模拟后端。
- 存在性指标分母、无效正例分布、分类 unknown 和宏平均计算。
- 原始 count 入口参数转发、原评估口径样例、30 个保护文件 SHA256 校验。

运行方式：

```bash
python -m unittest discover -s tests -v
python scripts/check_preserved.py
```

## 尚未执行

未在 AutoDL A800 上加载真实 Qwen3.5 9B，未对你们的 750 张实际样本跑全量识别。
没有据模拟模型的测试速度推断 GPU 吞吐，也没有宣称提高识别准确率。
当前环境验证了 Windows 文件锁分支；Linux fcntl 分支应在服务器实际运行时验证。
所有 GPU 相关模型兼容性、加速 kernel、显存、实际 JSON 有效率和准确率以服务器结果为准。

## 服务器最小验收

1. 继续使用已经跑通 count 的环境，运行 `scripts/show_environment.py` 保存实际版本。
2. 运行 `python scripts/check_preserved.py` 与 `python -m aquaroute doctor`。
3. 将已有增强图 prepared 文件用于 `infer --limit 16 --batch-size 8`，检查 `errors` 和 `fallback batches`。
4. 抽查原始回答与输入图片，确认识别目标、真值和模式正确。
5. 用同一个 run-id 执行 `infer --resume`，确认不重复计算已完成条件。
6. 全部完成后评估，与原图及 identity 条件做配对对比。
