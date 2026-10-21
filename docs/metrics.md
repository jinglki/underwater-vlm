# 识别指标说明

## 存在性

`N` 为有标签图片数，`V` 为返回有效结构的图片数。没有标签的样本只用于预测，不进入指标分母。
`P/Nneg` 分别为全部真实正例与负例；其中包括模型返回无效结构的图片。

| 字段 | 分母与含义 |
|---|---|
| accuracy | `(TP+TN)/N` 无效回答视为失败 |
| valid_accuracy | `(TP+TN)/V` 仅有效回答 |
| precision | `TP/(TP+FP)` |
| recall | `TP/(TP+FN)` 仅有效回答中的真实正例 |
| f1 | `2TP/(2TP+FP+FN)` 有效回答集合 |
| recall_all | `TP/P` 无效正例也视为失败 |
| specificity_all | `TN/Nneg` 无效负例也视为失败 |
| f1_all | `2TP/(2TP+FP+FN+invalid_positive)` 无效正例计为漏检 无效负例不是正例预测 |
| balanced_accuracy_all | `recall_all` 与 `specificity_all` 的均值 两类必须都有样本 |
| valid_response_rate | `V/N` |
| majority_baseline | `max(P,Nneg)/N` 永远回答多数类别的准确率 |

分母为零时返回 `null`。`error_rate=1-accuracy`，不是单独的误报率。
`confusion_matrix` 中的 TN/FP/FN/TP 仅计有效回答，`invalid` 计全部无效回答。
额外的 `invalid_positive/invalid_negative` 还原了无效回答对应的真实类别。

## 候选类别分类

每张图只有一个真实类别；预测可以为某个候选类别或 `unknown`。合法的 `unknown` 计为有效回答，但在所有真实候选类别上属于错误。
`confusion_matrix` 行为真实类别，列为候选类别加 unknown。无效回答单独计入 `invalid_by_class`。
每类的 `recall_all/f1_all` 把该类无效回答和 unknown 都计为未识别成功。
`macro_f1_all` 对当前条件中真实 support 大于零的类别做等权平均；零 support 类别不参与平均，但仍展示其统计。

## 与历史 count 的关系

count 的 `mama/evaluation.py` 没有改变，其 count_mae 仅在有效回答上取平均，计数全对率以全部有标签图片为分母。
新识别指标没有 count 或位置评分。不能把新识别任务与旧 count 任务的指标差异全部归因于增强或批次，二者提示词和输出要求不同。
多个增强条件是同一批样本的配对结果；统计显著性需使用逐条对齐的预测，不将各条件当独立数据。
