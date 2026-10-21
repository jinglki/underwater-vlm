# AquaRoute

面向水下图像增强的多模态识别实验项目。保留原有 `prepare → infer → evaluate` 框架，比较原图与不同增强条件下的结果。

**已有 count 实验保留，新识别流程独立运行。** `mama/` 中的 count 代码和增强代码均按原文件保留；识别使用 `aquaroute/`，默认只回答指定目标是否存在，也提供候选类别分类模式。

本次已确认的正式识别任务是**判断指定目标是否存在**。直接按[存在性识别运行指南](docs/quickstart_presence.md)操作即可；分类配置是可选示例，不参与默认实验。

## 1 从哪里开始

| 需要完成的工作 | 入口 |
|---|---|
| 继续或复现已完成的 count 实验 | `python -m mama ... --config config/count.yaml` |
| 判断指定 target 是否存在 | `python -m aquaroute ... --config config/recognition.yaml` |
| 从候选列表识别一个主要类别 | `python -m aquaroute ... --config config/classification.yaml` |
| 核对 count 与增强文件没有变化 | `python scripts/check_preserved.py` |
| 环境与资源检查 | `python -m aquaroute doctor` |

所有下述命令都在解压后的 **AquaRoute 根目录**运行。不要进入 `runs/` 再执行 `python -m ...`。

```text
AquaRoute/
├── aquaroute/                新识别入口 数据 模型 批次 断点 指标
├── mama/                     原始 count 与增强编排 不修改
├── config/
│   ├── recognition.yaml      Qwen3.5 9B 存在性识别 默认批次 8
│   ├── classification.yaml   候选类别识别示例
│   ├── count.yaml            原始 count 流程的 9B 配置
│   └── config.yaml           原始 4B 配置快照 原样保留
├── enhance/waternet/          原始 WaterNet 源码 权重 许可证
├── dataset/                  JSONL 清单与图片
├── models/                   自行放入完整的 Qwen 模型目录
├── img/enhance/               prepare 输出 按 run-id 存放
├── runs/                     prepared 预测与识别断点
├── results/                  按条件汇总的指标
├── scripts/                  校验与环境信息工具
├── tests/                    不依赖模型权重的回归测试
└── docs/                     架构 迁移 指标与验证说明
```

## 2 环境准备

### 已经跑通 count 的 AutoDL 环境

优先继续使用该环境，不要为了整理代码升级 torch、CUDA 或 Transformers。先记录实际版本：

```bash
cd /你的路径/AquaRoute
python scripts/show_environment.py > environment-server.json
python scripts/check_preserved.py
python -m aquaroute --help
python -m aquaroute doctor --config config/recognition.yaml
```

缺少轻量依赖时再安装：

```bash
python -m pip install -r requirements.txt
```

`doctor` 不加载模型权重，不会发起下载；它检查包、CUDA、Qwen 模型类、模型目录及 WaterNet 资源。若模型还未复制完整，出现失败项是预期结果。

### 新环境

推荐 Python 3.12。先根据显卡和驱动安装匹配的 [PyTorch 与 torchvision](https://pytorch.org/get-started/locally/)，再执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-inference.txt
python -m aquaroute doctor
```

Qwen3.5 需要安装版本确实包含 `Qwen3_5ForConditionalGeneration`；以 `doctor` 的实际检查和 [官方模型说明](https://huggingface.co/Qwen/Qwen3.5-9B) 为准。本项目没有从源码猜测你们服务器的完整锁定环境，`requirements` 也不是已验证的 CUDA 版本锁文件。

可选安装命令入口：`python -m pip install -e . --no-deps`，之后可用 `aquaroute ...`。部署采用解压目录或 editable 安装，配置与增强资源留在项目根目录，不把普通 wheel 当作独立资源包部署。

`causal-conv1d` 是可选加速依赖。你们已有的匹配 wheel 可继续使用；不将其设为强制安装项，避免构建或 GitHub 网络超时。若 CUDA、torch、Python 或 ABI 变化，原 wheel 可能不再适用。

## 3 模型与增强资源

项目包含 `enhance/waternet/`，无需再次 clone。新配置将权重指向随包保留的：

```text
enhance/waternet/waternet_exported_state_dict-daa0ee.pt
```

将已有完整模型目录复制或软链接到 `models/Qwen3.5-9B/`。该目录应包含模型配置、全部权重分片、tokenizer 与 processor/chat template 文件。9B 大模型权重不包含在发布 ZIP 中。

需要下载时选择一种方式：

```bash
python -m aquaroute download --config config/recognition.yaml --source huggingface
# 或先安装 modelscope 后使用：
python -m aquaroute download --config config/recognition.yaml --source modelscope
```

原始 `config/config.yaml` 保留了 4B 路径和 `mdoels` 拼写，作为历史快照不推荐直接使用。新增配置修复资源引用，增强算法、强度、尺寸和 Zero-DCE 步数不作改变。

## 4 数据格式

### 存在性识别 默认模式

每行一个样本，图片路径相对于 JSONL 所在目录。每个样本可包含多张图片，真值按图片顺序填写：

```json
{"id":"urchin-001","images":["images/a.jpg","images/b.jpg"],"target":"海胆","question":"图中是否存在海胆？只判断是否存在。","truth":[{"present":true},{"present":false}]}
```

也可复用 count 数据中的 `truth: [{"count":2},{"count":0}]`，识别评估将其解释为存在与不存在，不进行计数评分。若同时提供 `present` 与 `count`，两者必须一致。没有标签可省略 `truth`，仍输出预测，标签指标为 `null`。

模型需要返回的协议为：

```json
{"images":[{"image_index":0,"present":true},{"image_index":1,"present":false}]}
```

不要求 `count` 或 `locations`，避免为了判断存在性额外生成计数与位置文本。

### 候选类别识别

先修改 `config/classification.yaml` 的 `recognition.labels` 为实际类别。每张图片只选择一个主要类别，不能用于穷举图片中的所有物体：

```json
{"id":"class-001","images":["images/a.jpg"],"target":"主要海洋生物","question":"图中主要目标属于哪个候选类别？","truth":[{"label":"海胆"}]}
```

模型返回 `{"images":[{"image_index":0,"label":"海胆"}]}`；无法辨认或不属于候选类别时返回 `unknown`。`unknown` 是保留的预测值，不能放进真实候选类别表。

`dataset/examples/` 提供格式示例，引用的真实图像需自行放入 `dataset/images/`。示例中的物体与标签不是对你们实际数据的标注。

运行前可执行不加载模型的格式与路径检查：

```bash
python -m aquaroute validate --config config/recognition.yaml --manifest dataset/presence.jsonl
# 或检查已有增强图的条件、标签和路径：
python -m aquaroute validate --config config/recognition.yaml --prepared runs/presence_prepare/prepared.jsonl
```

原始的 `dataset/dataset_download.py` 是空占位文件，原样保留。本包没有根据未知的数据源编造下载器，请自行准备真实图片与标注。

## 5 识别实验 推荐分阶段运行

### 第一步 生成增强图

```bash
python -m aquaroute prepare \
  --config config/recognition.yaml \
  --manifest dataset/presence.jsonl \
  --run-id presence_prepare
```

每张图生成原图条件及 8 个槽位 × 3 档强度，共 25 个条件。其中 `reserved_identity` 是不改变内容的对照，不能算作真实有效增强方法。
`zero_dce` 保留原实现的逐图随机初始化与优化，`waternet` 保留原实现的加载行为。没有修改这些算法来提高准备速度。

### 第二步 小规模测试

```bash
python -m aquaroute infer \
  --config config/recognition.yaml \
  --prepared runs/presence_prepare/prepared.jsonl \
  --run-id presence_9b --batch-size 8 --limit 16
```

观察末尾的秒数、`errors` 与 `fallback batches`，并抽查 `runs/presence_9b/predictions.jsonl`。`--limit 16` 表示本次最多新增 16 个条件。它不是准确率实验，也不代表所有图片都能在相同显存内运行。

### 第三步 接着跑全部条件

```bash
python -m aquaroute infer \
  --config config/recognition.yaml \
  --prepared runs/presence_prepare/prepared.jsonl \
  --run-id presence_9b --batch-size 8 --resume
```

前面完成的结果不会重复。之后中断也使用同一条命令恢复。若显存不足，可将 `8` 改为 `6` 或 `4`；运行中 OOM 也会自动拆分并降低后续批次。每条请求仍独立，顺序不变。

### 第四步 评估

```bash
python -m aquaroute evaluate \
  --config config/recognition.yaml \
  --predictions runs/presence_9b/predictions.jsonl \
  --run-id presence_9b
```

结果为 `results/presence_9b_metrics.json`。有配套运行元数据时，不允许把未完成的识别运行误当作全量结果评估。

首次运行也支持一条命令：

```bash
python -m aquaroute all --config config/recognition.yaml \
  --manifest dataset/presence.jsonl --run-id presence_all
```

`all` 中断后不要重跑增强步骤；改用生成的 `prepared.jsonl` 执行 `infer --resume`，结束后再 `evaluate`。
分类实验将上述配置换成 `config/classification.yaml`，并使用相应标签清单。

## 6 直接复用已经生成的增强图

已有 `runs/a910a6f3be90/prepared.jsonl` 时，直接使用识别 `infer`，不需要再次运行增强。旧 count 的 `truth.count` 可以用于存在性评估，但旧 count 的预测结果不是新识别输出。

准备文件中的绝对路径如果仍指向原服务器且图片存在，可以直接读取；复制增强图后使用明确的前缀映射：

```bash
python -m aquaroute infer --config config/recognition.yaml \
  --prepared /你的旧运行/prepared.jsonl --run-id migrated_presence \
  --path-map /autodl-fs/data/mama-main=/autodl-fs/data/AquaRoute
```

映射只修改本次读取到的图片路径，不改磁盘上的旧 prepared 文件或图片。相对路径按 prepared 文件所在目录解析。迁移后应使用新 run-id；既有断点包含路径校验，不自动跨服务器合并。

## 7 count 实验保持原样

```bash
python -m mama prepare --config config/count.yaml --manifest dataset/count.jsonl --run-id count_prepare
python -m mama infer --config config/count.yaml --prepared runs/count_prepare/prepared.jsonl --run-id count_9b
python -m mama evaluate --config config/count.yaml --predictions runs/count_9b/predictions.jsonl --run-id count_9b
```

也可使用 `python -m aquaroute infer --task count --config config/count.yaml ...`，内部直接调用原始 count CLI。
原 `VisionModel.predict`、`infer(prepared,cfg)`、提示词、解析、评估、输入输出字段均未改。
因此历史 count 路径仍逐条推理、全部结束才保存，不支持新增的 batch/resume/limit 参数。识别的新速度和断点能力不会自动改变历史 count 行为。
若只想保留已完成的 count 结果，无需重跑或转换。

## 8 输出与指标

识别运行目录：

```text
runs/presence_9b/
├── predictions.jsonl   每批追加 原始回答 解析结果 错误信息
├── run.json            参数 输入校验 环境 完成状态
└── .aquaroute.lock     运行锁 防止两份识别任务写同一目录
```

无效 JSON、OOM 后单条仍失败等会记录为 `prediction: null` 和明确的 `error`，不会当作“目标不存在”。错误结果也计入完成数，恢复时不会自动重试它们。

存在性指标同时报告：全样本准确率、有效样本准确率、有效样本 precision/recall/F1、含无效回答失败的 `recall_all/f1_all`、平衡准确率、真实正负例数、无效回答类别分布与多数类基线。
这些是新的识别评估，**不覆盖历史 count 指标定义**。完整口径见 [指标说明](docs/metrics.md)。

## 9 验证与性能限制

```bash
python -m unittest discover -s tests -v
python scripts/check_preserved.py
```

自动测试使用小图片和模拟后端，验证协议、批次顺序、OOM、错误隔离、逐批保存、恢复、文件锁、CLI 与旧文件不变；不需要下载 9B 权重。
真实 A800 上的模型兼容性、吞吐与准确率必须按第 5 节的小规模任务实测，本包不承诺固定加速倍数或识别准确率提升。

进一步说明：

- [代码架构](docs/architecture.md)
- [迁移与改动边界](docs/migration.md)
- [指标定义](docs/metrics.md)
- [本次验证记录](docs/validation.md)

保留已有项目的 `LICENSE`。WaterNet 继续遵循其随附许可证；模型权重遵循对应模型许可。
