# 指定目标存在性识别运行指南

本次正式识别任务已确定为：对每张图片判断给定 `target` 是否存在。使用 `config/recognition.yaml`；不需要 `config/classification.yaml`，也不要求模型输出数量或位置。

## 1 保留旧实验

将 AquaRoute 解压到新目录，不覆盖已经完成 count 的 mama-main。继续使用已跑通的 Python 环境。
完整 Qwen3.5 9B 模型放到 `models/Qwen3.5-9B/`，或者将 `config/recognition.yaml` 的 `model.path` 改为服务器上已有模型的绝对路径。
不要为了这次代码整理升级原来的 CUDA、torch 或 Transformers。

在 AquaRoute 根目录检查：

```bash
python scripts/check_preserved.py
python -m aquaroute doctor --config config/recognition.yaml
```

## 2 使用已完成增强的图片

如果旧的准备文件和增强图片仍位于原目录，可以直接使用原文件的绝对路径。例如之前实验中的：

```bash
python -m aquaroute validate \
  --config config/recognition.yaml \
  --prepared /autodl-fs/data/mama-main/runs/a910a6f3be90/prepared.jsonl
```

路径以服务器实际位置为准。准备文件仍指向存在的原图与增强图时，无需再次 prepare。
如图片整体搬到了其他目录，在 infer 中用 `--path-map 旧前缀=新前缀`，具体见 README。

## 3 先测试 16 个条件

选择全新的 run-id，避免与旧 count 结果混淆：

```bash
python -m aquaroute infer \
  --config config/recognition.yaml \
  --prepared /autodl-fs/data/mama-main/runs/a910a6f3be90/prepared.jsonl \
  --run-id presence_9b --batch-size 8 --limit 16
```

检查日志中的 `errors` 和 `fallback batches`，并查看：

```bash
head -n 2 runs/presence_9b/predictions.jsonl
```

成功结果的 `prediction` 应类似：

```json
{"images":[{"image_index":0,"present":true}]}
```

`present` 必须是 JSON 布尔值 `true` 或 `false`，不是字符串。多图样本必须有同样数量、同样顺序的条目。
解析失败会保留 `raw` 和 `error`，不会默认为“不存在”。出现失败时先根据 error 排查，不把 16 条的测试当作完整实验。

## 4 继续全部条件

```bash
python -m aquaroute infer \
  --config config/recognition.yaml \
  --prepared /autodl-fs/data/mama-main/runs/a910a6f3be90/prepared.jsonl \
  --run-id presence_9b --batch-size 8 --resume
```

前 16 条将跳过。中断后继续使用这条命令；显存不足时可把 8 改为 6 或 4。
仅增大批次不保证成倍提速；严格比较结果时应固定批次与软件环境。
如果改了目标、提示词、模型、标签、识别代码或图片，应新建 run-id，不合并不同实验的断点。

## 5 全部结束后评估

```bash
python -m aquaroute evaluate \
  --config config/recognition.yaml \
  --predictions runs/presence_9b/predictions.jsonl \
  --run-id presence_9b
```

结果为 `results/presence_9b_metrics.json`。重点比较原图与各增强条件的 `f1`、`recall_all`、`balanced_accuracy_all`、`valid_response_rate`，同时看多数类基线，不只比较准确率。

旧 `truth.count` 可直接转换为存在性真值：大于零表示存在。只有识别提示词和输出协议发生变化，旧 count 的已完成结果与评估文件不作修改。

## 6 新数据尚未准备增强图时

先准备每行一个样本的清单，图片路径相对于清单文件：

```json
{"id":"urchin-001","images":["images/a.jpg"],"target":"海胆","question":"图中是否存在海胆？","truth":[{"present":true}]}
```

然后运行：

```bash
python -m aquaroute validate --config config/recognition.yaml --manifest dataset/presence.jsonl
python -m aquaroute prepare --config config/recognition.yaml --manifest dataset/presence.jsonl --run-id presence_prepare
```

将第 3 和第 4 步的 `--prepared` 改为 `runs/presence_prepare/prepared.jsonl`。
增强阶段保持原实现，仍会生成 25 个条件；其中三个 identity 条件是对照，不是新的增强算法。
