# AquaRoute 代码导览

## 框架与职责

原有 `download → prepare → infer → evaluate` 阶段、根目录布局、JSONL 数据流全部保留。
新增识别路径与冻结的 count 路径分开，防止新协议或新指标改变已完成的 count 实验。

```text
python -m aquaroute
        │
        ├─ task=count ── mama/__main__.py ── 原始 count 流程
        │
        └─ task=recognition
             ├─ config.py       读取和校验 YAML
             ├─ data.py         清单与标签校验 按样本和条件展开
             ├─ mama/prepare    复用原有预处理与增强编排
             ├─ protocol.py     presence 或 classification 的提示词与解析
             ├─ model.py        一份模型 权重一次加载 图片并发读取 GPU 批量生成
             ├─ engine.py       顺序批次 OOM 拆分 错误隔离
             ├─ storage.py      文件锁 每批保存 输入校验 断点恢复
             └─ metrics.py      识别指标与无效回答分布
```

## 输入与结果边界

- 一条请求对应一个样本的一个增强条件，可包含多张图片。
- 批次只是同时送入模型的多条独立对话；不把不同条件拼进同一个用户问题。
- 多图样本的 `images` 和 `truth` 顺序固定；返回的 `image_index` 必须连续。
- 结果顺序等于输入样本顺序与 `conditions` 插入顺序。OOM 拆分也不改变顺序。
- 每个预测保留 `id/target/question/condition/images/truth/raw/prediction/error`，识别另附 `task/recognition_mode`。
- count 的字段、提示词、解析器、评估函数与原始版本一致。

## 性能与边界

GPU 通过一次 `generate` 处理多个独立请求提高吞吐，不创建多份模型，不在多个 Python 线程里同时调用同一模型的 `generate`。
CPU 线程仅解码图片，内存中最多保留当前批次。有效批次上限由显存、图片数量、分辨率和实际生成长度决定。
识别的 JSON 协议不要求 count 与 locations，因此通常需要生成更少 token，但实际速度必须在服务器测量。
本项目不修改 WaterNet、Zero-DCE 或其他增强算法，也不通过改尺寸、改强度、替换算法来制造提速。

识别断点使用文件级别的运行锁。每批 append、flush、fsync 后才计入完成；断点按结果文件重建，`run.json` 中的计数仅为辅助状态。
未完成的末尾行会截断并重算；已换行但无法解析的记录会报错，不会悄悄跳过。
同一运行的失败记录也视为已完成，恢复时不会自动重试它们，避免无限重跑。

恢复校验涵盖输入内容、关键识别源码、模型配置、运行库版本、模型文件大小和修改时间，以及输入图像大小和修改时间。
它不是对多 GB 权重与全部图像内容的加密完整性证明。即使保留时间戳替换了内容，也必须使用新的 run-id。
批次大小和图片读取线程数可在恢复时调整；浮点数批处理不保证与逐条处理得到逐字相同的结果。严谨复验应固定批次与环境。

## 原始文件保护

`mama/*.py`、`config/config.yaml` 以及 `enhance/` 中保留的文件使用 SHA256 校验。
`enhance.zip` 中的嵌套 `.git` 元数据和 `__pycache__` 编译缓存不进入发布包；源码、权重、许可证、说明和示例资源保持原字节。
运行 `python scripts/check_preserved.py` 可检查。新增配置只修复资源引用，不更改增强参数或增强源码。

## 结构参考

借鉴以下项目按配置、数据、模型和评估职责组织代码的方式；未复制其训练代码，也未引入 Trainer 或 Lightning 框架。

- [Transformers 图像分类示例](https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py)
- [Transformers 多模态接口](https://huggingface.co/docs/transformers/chat_templating_multimodal)
- [Processor 参数分发实现](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py)
- [Qwen3.5 9B 模型说明](https://huggingface.co/Qwen/Qwen3.5-9B)

WaterNet 原始项目及许可证随 `enhance/waternet/` 保留。
