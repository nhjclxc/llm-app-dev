# PyTorch 损失函数：从 0 到 1

损失函数（loss function / criterion）把模型输出和真实目标之间的差异变成一个可优化的标量。本文从误差和梯度的直觉开始，逐步学习 PyTorch 的损失 API、常见任务选择、张量形状、加权与 reduction、多任务损失、完整训练流程和排错方法。

示例默认：

```python
import torch
from torch import nn
```

不同 PyTorch 版本的参数细节可能略有变化；实际项目请对照当前版本的官方文档。

## 1. 学习路线与先修知识

| 阶段 | 要回答的问题 | 本文内容 |
| --- | --- | --- |
| 入门 | 什么是损失？它怎样影响参数？ | 2–3 |
| 基础 API | `nn.Module` 损失函数如何调用？ | 4–5 |
| 任务选择 | 回归、分类、分割、序列任务用什么？ | 6–10 |
| 进阶实践 | 权重、掩码、多个损失怎样处理？ | 11–13 |
| 完整训练 | 怎样验证形状、反传和评估？ | 14–16 |

建议先理解 Tensor 的 shape/dtype、`nn.Module`、autograd 的 `backward()` 和 optimizer 的更新流程。损失函数既不是优化器，也不负责更新参数：

```text
输入 x -> model(x) 得到预测 -> loss(prediction, target) -> backward 得到梯度 -> optimizer.step 更新参数
```

## 2. 什么是损失函数

模型参数记为 `theta`，模型预测记为 `y_hat = f_theta(x)`，真实目标为 `y`。损失函数衡量预测与目标之间的差异：

$
L = \ell(\hat{y}, y)
$

训练的目标通常是通过优化器调整参数，使训练数据上的损失降低。对于一个 batch，损失一般先对样本求和或平均，得到标量，再调用 `loss.backward()` 计算梯度。

损失值不是准确率，也不一定能直接解释为“错了百分之多少”。它只是训练目标的数值；分类准确率等指标需要单独计算。损失越小通常表示模型更符合该损失的定义，但不保证验证集效果更好，也不保证不同损失之间的数值可以直接比较。

## 3. 从一个简单误差开始

假设真实值为 3，模型预测为 2：

```python
target = torch.tensor([3.0])
prediction = torch.tensor([2.0], requires_grad=True)
loss = (prediction - target).pow(2).mean()
print(loss)  # tensor(1., grad_fn=<MeanBackward0>)
loss.backward()
print(prediction.grad)  # tensor([-2.])
```

梯度为负，梯度下降沿负梯度的反方向更新预测对应的参数，因此预测倾向于增大。模型训练时，梯度会从 loss 经计算图继续传到产生 prediction 的模型参数。

这段手写计算只适合理解机制。实际训练优先使用 PyTorch 提供的损失模块，它们封装了维度约定、权重、忽略标签和数值稳定实现。

## 4. 损失函数的两种调用方式

### 4.1 模块形式：推荐用于常规训练

```python
criterion = nn.MSELoss()
prediction = torch.tensor([1.0, 2.0], requires_grad=True)
target = torch.tensor([1.5, 2.5])
loss = criterion(prediction, target)
```

损失模块是 `nn.Module`，可以作为训练代码中的 `criterion` 使用。含类别权重的损失可以把权重作为 buffer 注册在模块中，随模块迁移设备。

### 4.2 函数形式

```python
import torch.nn.functional as F

loss = F.mse_loss(prediction, target)
```

`torch.nn.functional` 中的函数适合在 `forward()` 内组合自定义计算，或临时改变参数。模块和函数通常提供相同的数学目标；选择主要看是否需要保存配置、注册 buffer、复用模块对象。

## 5. 最重要的通用参数：`reduction`

许多损失模块都支持 `reduction`：

| 取值 | 含义 | 常见用途 |
| --- | --- | --- |
| `"mean"` | 对参与计算的元素/项求平均，输出标量 | 默认训练基线 |
| `"sum"` | 求和，输出标量 | 需要总量目标或自行归一化 |
| `"none"` | 不做最终归约，保留逐项损失 | mask、样本权重、自定义归一化 |

不同损失对“参与平均的项”的定义不完全相同。例如 `CrossEntropyLoss` 有类别权重、`ignore_index` 时，其 `mean` 按有效项及权重规则归一化；不要假设所有损失都严格除以 batch size。查看对应 API 文档，并在小例子上验证。

```python
criterion = nn.L1Loss(reduction="none")
per_element = criterion(torch.tensor([1.0, 4.0]), torch.tensor([2.0, 1.0]))
print(per_element)  # tensor([1., 3.])
print(per_element.mean())  # tensor(2.)
```

`mean` 会让 loss 的尺度通常不随 batch 中元素数量线性增长；`sum` 会使 batch 大小时损失和梯度尺度变化。梯度累积或分布式训练时，必须弄清楚归一化发生在哪里。

## 6. 先确定任务，再确定损失

损失函数选择不能只看名字。首先回答：

1. 目标是连续数值、类别、多个独立标签、像素类别，还是 token 序列？
2. 每条样本有一个标签，还是每个位置都有标签？
3. 模型最后一层输出的是原始分数（logits）、概率，还是连续预测？
4. 标签的形状和 dtype 是什么？哪些样本/位置需要忽略？
5. 错误代价是否不对称？类别是否不平衡？

常用选择速查：

| 任务 | 模型输出 | target 常见格式 | 推荐起点 |
| --- | --- | --- | --- |
| 单目标/多目标回归 | 任意实数 | 浮点数，形状与输出相同 | `MSELoss` 或 `L1Loss` |
| 回归且关注异常值鲁棒性 | 任意实数 | 浮点数 | `SmoothL1Loss` / `HuberLoss` |
| 二分类 | 每样本一个 logit | 0/1 浮点数，形状与 logit 相同 | `BCEWithLogitsLoss` |
| 多标签分类 | 每类别一个 logit | 多个独立 0/1 浮点标签 | `BCEWithLogitsLoss` |
| 单标签多分类 | 每类别一个 logit | 类别索引 `long` | `CrossEntropyLoss` |
| 多分类且标签为软分布 | 每类别一个 logit | 与 logits 同形状的概率分布 | `CrossEntropyLoss` 概率目标或 KL 散度，依场景选择 |
| 逐像素分割 | 每像素每类别一个 logit | 类别索引图 `long` | `CrossEntropyLoss` |
| 二值分割 | 每像素一个 logit | 0/1 浮点 mask | `BCEWithLogitsLoss` |
| 语言模型 next-token | 每 token 每词表一个 logit | token 类别索引 `long` | `CrossEntropyLoss(ignore_index=...)` |

回归：MSELoss
二分类：BCEWithLogitsLoss
多分类、语言模型：CrossEntropyLoss

这个表是起点而非定律。目标分布、代价函数和数据标注方式决定最终选择。

## 7. 回归损失

### 7.1 均方误差 `MSELoss`

$
L_{MSE}=\frac{1}{N}\sum_i(\hat y_i-y_i)^2
$

```python
criterion = nn.MSELoss()
prediction = torch.tensor([[2.0], [4.0]], requires_grad=True)
target = torch.tensor([[3.0], [5.0]])
loss = criterion(prediction, target)    
```

误差平方会放大较大的偏差，因此适合误差平方有明确意义、噪声近似高斯的回归基线。缺点是异常值可能主导训练。

### 7.2 平均绝对误差 `L1Loss`

\[
L_{L1}=\frac{1}{N}\sum_i|\hat y_i-y_i|
\]

```python
criterion = nn.L1Loss()
```

对异常值比 MSE 更稳健，但在误差为零处不可微；PyTorch 使用可用于优化的次梯度。

### 7.3 `SmoothL1Loss` 与 `HuberLoss`

小误差区域近似平方误差，大误差区域近似绝对误差：

```python
criterion = nn.SmoothL1Loss(beta=1.0)
# torch.nn.HuberLoss(delta=1.0)  # 使用时查当前版本的参数定义
```

目标和预测的量纲会影响阈值参数 `beta` / `delta`。归一化目标后，默认阈值可能意味着不同的误差范围。目标值尺度变化时应重新检查。

### 7.4 回归例子与形状

```python
model = nn.Linear(4, 1)
criterion = nn.MSELoss()
x = torch.randn(8, 4)
target = torch.randn(8, 1)  # float；与输出 [8, 1] 同形状
prediction = model(x)
loss = criterion(prediction, target)
```

不要让 `[N, 1]` 的预测和 `[N]` 的标签意外广播成 `[N, N]`。如目标是每样本一个数，显式统一为 `[N]` 或 `[N, 1]`。始终检查 `prediction.shape`、`target.shape` 和 `dtype`。

## 8. 二分类与多标签：`BCEWithLogitsLoss`

Binary Cross Entropy 描述二元目标的概率似然。训练时推荐使用 `nn.BCEWithLogitsLoss`，它把 sigmoid 和 BCE 合并，以提高数值稳定性。

```python
model = nn.Linear(10, 1)  # 最后一层输出原始 logit，不加 Sigmoid
criterion = nn.BCEWithLogitsLoss()
x = torch.randn(4, 10)
target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
logits = model(x)
loss = criterion(logits, target)
```

约定：logits 和 target 通常形状相同；target 是浮点数，值在 `[0, 1]`。二分类标签也可以使用软目标，例如标签平滑或概率标注。

推理时才把 logits 转成概率：

```python
probabilities = torch.sigmoid(logits)
predicted_class = (probabilities >= 0.5).long()
```

阈值 `0.5` 是在等代价且概率校准合理时常见的默认值，并非所有业务的最佳阈值。

### 8.1 二分类与多标签的区别

- 二分类：一个样本属于正类或负类，输出 1 个 logit。
- 多标签：一个样本可以同时拥有多个标签，每个类别独立判断，输出 `C` 个 logits。
- 单标签多分类：一个样本只属于 `C` 个互斥类别之一，通常使用 `CrossEntropyLoss`，不是对每类独立做 BCE。

多标签例子：

```python
model = nn.Linear(20, 3)  # 三个标签可以同时为 1
criterion = nn.BCEWithLogitsLoss()
logits = model(torch.randn(5, 20))
targets = torch.tensor([
    [1., 0., 1.], [0., 1., 0.], [1., 1., 0.],
    [0., 0., 0.], [1., 0., 0.],
])
loss = criterion(logits, targets)
```

### 8.2 类别不平衡：`pos_weight`

正例远少于负例时，可使用 `pos_weight` 提高正例项的损失权重：

```python
pos_weight = torch.tensor([4.0])  # 输出有 C 个标签时，通常是形状 [C]
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

常见起点是训练集负例数/正例数，但这不是保证最优的规则。`pos_weight` 会改变精确率/召回率的权衡和概率校准；应在验证集上选择阈值、检查 PR 曲线，并且只用训练集统计权重，避免数据泄漏。

### 8.3 `BCELoss` 何时使用

`nn.BCELoss` 接收已经经过 sigmoid 的概率：

```python
prob = torch.sigmoid(logits)
loss = nn.BCELoss()(prob, target)
```

新代码训练一般优先 `BCEWithLogitsLoss(logits, target)`。不要把 logits 直接传给 `BCELoss`，也不要把 sigmoid 后的概率再传给 `BCEWithLogitsLoss`，否则会重复处理。

## 9. 单标签多分类：`CrossEntropyLoss`

`CrossEntropyLoss` 适合每个样本只有一个正确类别的互斥分类。对于类别数 `C`，模型输出每类一个原始 logit。

```python
model = nn.Linear(10, 3)  # [N, 3] logits
criterion = nn.CrossEntropyLoss()
x = torch.randn(4, 10)
target = torch.tensor([2, 0, 1, 2], dtype=torch.long)  # [N] 类别索引
logits = model(x)
loss = criterion(logits, target)
```

`CrossEntropyLoss` 内部结合 `LogSoftmax` 与负对数似然。训练时输入原始 logits：

```python
loss = criterion(logits, target)  # 正确
# 不要 criterion(torch.softmax(logits, dim=1), target)
```

若只需要预测类别，softmax 不改变最大值所在类别：

```python
predicted_class = logits.argmax(dim=1)
```

要展示概率时再做：

```python
probabilities = torch.softmax(logits, dim=1)
```

### 9.1 形状和 dtype 约定

| 场景 | logits | target（类别索引形式） |
| --- | --- | --- |
| 普通分类 | `[N, C]` | `[N]`，`torch.long` |
| 序列分类/逐 token | `[N, C, L]` | `[N, L]`，`torch.long` |
| 2D 逐像素分类 | `[N, C, H, W]` | `[N, H, W]`，`torch.long` |

类别维必须放在第 2 维（维度索引 1）。目标里是类别编号，不是 one-hot；标签范围为 `[0, C-1]`，被忽略的索引除外。

对 `[N, L, C]` 的 Transformer 输出，可将类别维移到第 2 维，或者展平 token 维：

```python
logits = torch.randn(2, 5, 7)  # [N, L, C]
target = torch.randint(0, 7, (2, 5), dtype=torch.long)
loss = nn.CrossEntropyLoss()(logits.transpose(1, 2), target)  # [N, C, L]

# 等价的展平写法
loss = nn.CrossEntropyLoss()(logits.reshape(-1, 7), target.reshape(-1))
```

需要忽略 padding token 时：

```python
pad_id = 0
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
```

忽略目标位置的损失和梯度。确认 `pad_id` 不会与真实类别冲突，并确保平均损失按有效 token 而非总 token 解释。

### 9.2 软标签、概率目标和 label smoothing

当前 PyTorch 的 `CrossEntropyLoss` 支持类别索引目标，也支持与 logits 同形状的概率目标；概率目标应是有效分布（每行非负且和为 1）。索引目标更高效，且能直接使用 `ignore_index`。软目标可用于 mixup、知识蒸馏或概率标注，但要确认当前版本 API 行为。

普通硬标签可用 label smoothing：

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

它把部分概率质量分给其他类别，可能改善过度自信，但不是通用增益；应在验证集上评估。不要同时误把 one-hot 浮点数组当作类别索引目标。

### 9.3 类别权重

类别不平衡时可为较少见或代价较高的类别设置权重：

```python
class_weights = torch.tensor([1.0, 2.0, 4.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

`weight` 的长度等于类别数，device 和 logits 一致。权重只能从训练数据或业务成本中确定；需要观察每类 precision/recall、混淆矩阵，不要只看总体 accuracy。加权后 loss 的标度及有效平均规则会变化。

## 10. 分割、序列与逐元素损失

### 10.1 多类图像分割

每个像素属于一个互斥类别时，模型输出 `[N, C, H, W]` logits，目标为 `[N, H, W]` 的 long 类别图：

```python
logits = torch.randn(2, 4, 64, 64)  # 4 类
mask = torch.randint(0, 4, (2, 64, 64), dtype=torch.long)
criterion = nn.CrossEntropyLoss(ignore_index=255)
loss = criterion(logits, mask)
```

目标 mask 不要为了交叉熵改成 one-hot（除非有意使用概率目标 API）。若有无效像素，给它们统一的 `ignore_index`，且该值不在有效类别编号内。

### 10.2 二值分割

前景/背景二分类常用单通道 logits `[N, 1, H, W]` 与同形状浮点 mask：

```python
logits = torch.randn(2, 1, 64, 64)
target = torch.randint(0, 2, (2, 1, 64, 64)).float()
loss = nn.BCEWithLogitsLoss()(logits, target)
```

若使用 `pos_weight`，其广播维度必须与需要加权的类别/通道对应。像素任务里广播规则尤其容易出错；先用小张量确认形状，不要靠自动广播猜测。

### 10.3 语言模型和序列标签

自回归语言模型通常对每个位置预测下一个 token，logits 末维为词表大小 `V`。将 logits reshape 为 `[N*L, V]`、标签 reshape 为 `[N*L]`，使用 `CrossEntropyLoss(ignore_index=pad_id)`。还需确保 logits 与 target 已按 next-token 目标正确错开一位；损失本身不会替你做移位。

### 10.4 需要自定义 mask 的损失

对回归任务屏蔽无效位置，可用 `reduction="none"` 后自行归一化：

```python
prediction = torch.randn(2, 3, requires_grad=True)
target = torch.randn(2, 3)
valid = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)

element_loss = nn.MSELoss(reduction="none")(prediction, target)
loss = element_loss[valid].mean()
```

若 mask 可能全为无效，直接 `.mean()` 会得到 NaN。生产代码要定义此情形的处理方式：跳过该 batch、返回可微的零损失，或用分母 clamp；并确保这种策略符合任务语义。

## 11. 样本权重、元素权重与 reduction

对每个样本给权重时，常见做法是保留逐样本 loss，再归一化：

```python
per_sample = nn.MSELoss(reduction="none")(prediction, target)
per_sample = per_sample.mean(dim=1)  # [N, D] 时先汇总每个样本的特征维
sample_weight = torch.tensor([1.0, 0.5, 2.0])
loss = (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
```

注意“按权重和归一化”与“加权后按样本数平均”不是同一个目标。对元素级权重，应让权重 shape 能明确广播到 element loss；必要时显式 reshape，避免静默广播错轴。

部分损失有内建权重：`CrossEntropyLoss(weight=...)`、`BCEWithLogitsLoss(weight=..., pos_weight=...)`。阅读参数文档区分：

- `weight`：通常用于逐元素/逐样本的损失缩放，必须满足该 API 的广播约定。
- `pos_weight`：BCE 中专门缩放正例项；不是普通样本权重。
- `class weight`：交叉熵按目标类别缩放。

## 12. 不同目标不要混为一谈

### 12.1 KL 散度 `KLDivLoss`

KL 散度常用于分布对齐或知识蒸馏。PyTorch 的 `nn.KLDivLoss` 输入约定要特别注意：input 通常是 log-probabilities，target 是概率分布；使用 `log_target=True` 时 target 也按 log 概率解释。

```python
student_log_prob = torch.log_softmax(student_logits, dim=-1)
teacher_prob = torch.softmax(teacher_logits, dim=-1)
loss = nn.KLDivLoss(reduction="batchmean")(student_log_prob, teacher_prob)
```

默认 `reduction="mean"` 可能不是概率理论中通常期待的按 batch 归一化；蒸馏常见 `batchmean`。含温度 `T` 的蒸馏还要按方法约定缩放（常见乘 `T**2`），不要在没有明确目标时混用。

### 12.2 排序、度量学习、对比学习

排序可以考虑 `MarginRankingLoss`；距离度量可考虑 `TripletMarginLoss`；对比学习常用基于 logits 的交叉熵构造。它们需要特定的正负样本组织方式和采样策略，不能仅靠换一个损失模块解决数据构造问题。先确认样本对/三元组/正负样本定义，再核对 API 输入和 margin。

## 13. 多个损失组合

多任务模型或附加正则项常有总目标：

\[
L_{total}=\lambda_1L_1+\lambda_2L_2+\cdots+\lambda_kL_k
\]

```python
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.backbone = nn.Linear(input_dim, 32)
        self.classifier = nn.Linear(32, num_classes)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        hidden = torch.relu(self.backbone(x))
        return self.classifier(hidden), self.regressor(hidden)

class_loss_fn = nn.CrossEntropyLoss()
reg_loss_fn = nn.SmoothL1Loss()
model = MultiTaskModel(10, 3)
class_logits, value = model(torch.randn(8, 10))
class_targets = torch.randint(0, 3, (8,), dtype=torch.long)
value_targets = torch.randn(8, 1)
class_loss = class_loss_fn(class_logits, class_targets)
reg_loss = reg_loss_fn(value, value_targets)
total_loss = 1.0 * class_loss + 0.2 * reg_loss
```

损失的数值尺度和梯度大小可能不同。系数 `lambda` 应由任务重要性和训练动态决定；不能因为两个 loss 数字相近就假定它们对共享参数的梯度贡献相近。先分别记录各项 loss 和梯度行为，再做控制变量实验。

## 14. 完整训练与验证模板

下面用多分类展示标准顺序：

```python
model = nn.Linear(10, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(32, 10)
target = torch.randint(0, 3, (32,), dtype=torch.long)

model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(x)
loss = criterion(logits, target)
loss.backward()
optimizer.step()

print("loss:", loss.detach().item())
```

验证阶段不需要反向传播：

```python
model.eval()
with torch.inference_mode():
    logits = model(x)
    valid_loss = criterion(logits, target)
    prediction = logits.argmax(dim=1)
    accuracy = (prediction == target).float().mean()
```

`eval()` 控制 Dropout、BatchNorm 等模块行为；`inference_mode()` 关闭推理过程的 autograd 记录。模型的训练/验证模式与 loss reduction 是两个不同问题。

### 14.1 正确累计 epoch loss

若 `criterion` 是按 batch `mean`，不能简单对每个 batch 的 loss 做无权平均（最后一批可能更小）。对普通样本级任务可按 batch 样本数加权：

```python
total_loss = 0.0
total_samples = 0
for inputs, targets in loader:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    batch_size = inputs.size(0)
    total_loss += loss.detach().item() * batch_size
    total_samples += batch_size
epoch_loss = total_loss / total_samples
```

对于 `ignore_index`、类别权重、像素/token 任务，batch 中有效元素数和有效权重可能不同。若要精确的全数据集平均，应累计未归约 loss 和正确的分母，而非机械乘 batch size。

## 15. 选择损失的实用决策流程

1. **定义预测目标**：连续值、互斥类别、独立标签、概率分布、每像素/每 token 标签？
2. **确定模型输出语义**：回归原值；BCE/CE 通常输出 logits；推理展示概率时再做 sigmoid/softmax。
3. **写出 shape 与 dtype**：在代码里打印一次 `prediction.shape/dtype` 和 `target.shape/dtype`。
4. **从标准损失建立基线**：回归 `MSE` 或 `SmoothL1`；单标签分类 `CrossEntropy`；二分类/多标签 `BCEWithLogits`。
5. **先在小批数据过拟合**：确认 loss 能下降、梯度有限且目标对齐。做不到时先排查数据、标签、shape、激活和学习率，不要盲目换损失。
6. **分析业务错误代价**：异常值、类不平衡、漏报与误报代价、无效标签，分别考虑鲁棒损失、权重、阈值或 mask。
7. **用验证集评估目标指标**：loss 不是唯一指标；分类任务看每类指标，回归看 MAE/RMSE 等，分割看 IoU/Dice 等。
8. **一次改变一个因素**：损失、权重、阈值、optimizer、增强策略不要同时改，保留实验记录。

## 16. 常见错误与排查清单

| 现象/报错 | 常见原因 | 检查方式 |
| --- | --- | --- |
| CE 报 target 类型错误 | 类别索引不是 `long`，或误用 one-hot | 打印 dtype；索引标签用 `long` |
| BCE 报类型错误 | target 是 `long` | BCE target 转浮点并确保在 `[0,1]` |
| loss 维度报错 | logits/target shape 不符合任务约定 | 在 loss 调用前打印 shape |
| loss 可运行但数值异常 | 静默广播、标签错位、错误类别维 | 对比逐样本/逐位置结果；检查广播和类别范围 |
| CE loss 很大或不收敛 | 对 logits 先做了 softmax，或标签映射不一致 | 直接传 raw logits；核对 class id |
| BCE 学习不稳定 | sigmoid 后又送入 `BCEWithLogitsLoss`，或反之 | 检查模型末层和 criterion 是否重复 sigmoid |
| 回归 loss 突然变大 | target 尺度差异/异常值，MSE 放大大误差 | 检查 target 统计量和归一化 |
| 验证 loss 与训练 loss 不可比 | reduction、mask、权重或有效 token 数不同 | 统一统计口径并记录有效分母 |
| loss 为 NaN | 空 mask、输入非有限、数值溢出或学习率过大 | 检查输入、loss、梯度 `isfinite` |
| loss 不下降 | 梯度图断开、目标错误、参数未更新 | 检查 `requires_grad`、梯度、`optimizer.step()` |

建议在训练初期加断言：

```python
assert torch.isfinite(logits).all()
assert torch.isfinite(loss)
assert logits.device == target.device
print(logits.shape, logits.dtype, target.shape, target.dtype)
```

排错时用一个极小 batch：让模型尝试记住这几条数据。若损失仍无法明显下降，优先检查 forward、标签、损失输入约定和梯度链路。

## 17. 学习练习

1. 手算两个样本的 MSE 与 L1，并用 PyTorch 对照。
2. 用同一二分类模型比较 `BCEWithLogitsLoss` 与错误的重复 sigmoid 写法，观察概率和梯度。
3. 构造 `[N, C]` logits 与 long 标签，完成多分类；再把输入换成 `[N, L, C]`，正确处理类别维。
4. 给分类任务添加 `ignore_index`，统计有效 token 数，验证平均 loss 的分母。
5. 用不平衡数据比较普通 BCE 与 `pos_weight`，同时报告 precision、recall 和 PR 曲线。
6. 在同一个回归任务上比较 MSE、L1、SmoothL1，加入异常值并观察敏感度。
7. 实现一个分类+回归多任务模型，记录各 loss 和总 loss，尝试调整权重。

完成标准：能在不查抄代码的情况下，说明目标格式、模型输出、损失的输入契约、reduction 规则，以及为什么该损失符合任务目标。

## 18. 关键结论

- 损失函数定义优化目标；autograd 算梯度，optimizer 更新参数。
- 选损失首先看任务语义、输出和标签格式，不是比较名字或数值大小。
- 回归常从 MSE / SmoothL1 起步；单标签多分类用 raw logits + `CrossEntropyLoss`；二分类和多标签用 raw logits + `BCEWithLogitsLoss`。
- `reduction`、类别权重、mask 和 `ignore_index` 会改变统计口径，日志累计也要与其一致。
- 能下降的训练 loss 不等于好模型；最终以验证集和任务指标选择方案。

## 19. 参考资料

- [PyTorch `torch.nn` 官方文档](https://docs.pytorch.org/docs/stable/nn.html)
- [PyTorch 损失函数官方文档](https://docs.pytorch.org/docs/stable/nn.html#loss-functions)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
