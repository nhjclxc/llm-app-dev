# MLP 多层感知机：从前向传播到正则化

多层感知机（Multi-Layer Perceptron，MLP）由若干全连接层和非线性变换组成，可以处理表格特征、向量表示和展平后的图像。本文从单个神经元开始，推导前向传播和反向传播，再学习激活函数、梯度稳定性、参数初始化、正则化、训练流程与排错。

示例使用 PyTorch 2.x 风格 API：

```python
import torch
from torch import nn
```

## 1. 学习地图与先修知识

| 阶段 | 学习目标 | 内容 |
| --- | --- | --- |
| 入门 | 认识神经元、层、网络和张量形状 | 2–4 |
| 核心计算 | 理解前向传播、损失和反向传播 | 5–8 |
| 稳定训练 | 学习激活、梯度问题、初始化与归一化 | 9–13 |
| 泛化能力 | 理解过拟合与正则化 | 14–16 |
| 实战 | 搭建分类/回归模型并诊断训练 | 17–20 |

建议先熟悉 Tensor 基本操作、`nn.Module`、`nn.Linear`、autograd、损失函数和 optimizer。MLP 的核心链路是：

```text
输入 -> 前向传播得到预测 -> 损失函数比较预测和目标 -> 反向传播求梯度 -> 优化器更新参数
```

## 2. 从线性模型到神经元

一个线性模型对输入特征做加权求和：

$
z = w_1x_1+w_2x_2+\cdots+w_dx_d+b = \mathbf{w}^T\mathbf{x}+b
$

`w` 是权重，控制每个输入特征对结果的影响；`b` 是偏置，允许决策边界不必经过原点。线性模型只能形成线性决策边界。

神经元通常再对线性结果应用激活函数 `phi`：

$
a=\phi(z)=\phi(\mathbf{w}^T\mathbf{x}+b)
$

激活函数引入非线性，使多层组合后能够表达复杂的函数关系。若移除所有非线性，无论堆多少个线性层，整体仍等价于一个线性变换。

## 3. MLP 的层与张量形状

全连接层 `nn.Linear(in_features, out_features)` 对输入最后一维做仿射变换。对批次输入：

```text
输入 x:  [N, D_in]
权重 W: [D_out, D_in]
偏置 b:  [D_out]
输出 y:  [N, D_out]
```

PyTorch 实际使用 `y = x @ W.T + b`。一个三层 MLP 可以是：

```text
[N, D_in] -> Linear(D_in, H1) -> 激活
           -> Linear(H1, H2)   -> 激活
           -> Linear(H2, D_out)
```

`N` 是 batch size，`D_in` 是每条样本的输入特征数，`H1/H2` 是隐藏层宽度，`D_out` 由任务决定。分类输出层通常直接输出 logits；回归输出层通常输出连续实数。

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)
x = torch.randn(8, 10)
print(model(x).shape)  # [8, 3]
```

## 4. 为什么需要隐藏层和非线性

单个线性层无法解决异或（XOR）问题：XOR 的两类点无法由一条直线分开。隐藏层通过把输入映射到新的表示空间，再使用非线性决策边界解决此类非线性关系。

两层线性变换若中间没有激活：

$
W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)
$

仍只是一个线性变换。因此实用的 MLP 通常在隐藏层后加入 ReLU、GELU、Tanh 等激活函数。输出层是否使用激活取决于损失函数和任务约定，不应机械地每层都加激活。

## 5. 前向传播：逐层计算预测

前向传播（forward propagation）是把输入按网络定义逐层变换，得到模型输出的过程。对含 `L` 层的网络：

$
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)},\qquad
a^{(l)}=\phi^{(l)}(z^{(l)})
$

其中 `a^(0)=x`；隐藏层一般经过激活，最后一层的处理由任务决定。举例：

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        hidden = self.activation(self.fc1(x))
        return self.fc2(hidden)

model = MLP(10, 32, 2)
logits = model(torch.randn(4, 10))
print(logits.shape)  # [4, 2]
```

调用 `model(x)` 会进入模块调用机制并执行 `forward()`。层应在 `__init__` 中创建并注册；若在 `forward()` 中临时创建 `nn.Linear`，其参数不能像稳定注册的模型参数那样被持续优化。

## 6. 输出层、logits 和任务选择

隐藏层结构可以相似，输出层和损失函数要按任务匹配：

| 任务 | 输出层宽度 | 模型输出 | 常用损失 | target 典型形状/dtype |
| --- | ---: | --- | --- | --- |
| 单目标回归 | 1 | 任意实数 | `MSELoss` / `L1Loss` | `[N, 1]` 浮点 |
| 单标签多分类 | 类别数 `C` | `C` 个 logits | `CrossEntropyLoss` | `[N]`，`long` 类别索引 |
| 二分类 | 1 | 每样本一个 logit | `BCEWithLogitsLoss` | `[N, 1]` 浮点 0/1 |
| 多标签分类 | 标签数 `C` | `C` 个独立 logits | `BCEWithLogitsLoss` | `[N, C]` 浮点 0/1 |

`CrossEntropyLoss` 内部处理 log-softmax，训练时不要在模型最后先加 softmax。`BCEWithLogitsLoss` 内部结合 sigmoid 和二元交叉熵，训练时不要先加 sigmoid。推理时才按需求转换为概率。

```python
logits = model(x)                         # 多分类 [N, C]
probabilities = torch.softmax(logits, dim=1)
predicted_class = logits.argmax(dim=1)

binary_logits = binary_model(x)           # 二分类 [N, 1]
binary_probabilities = torch.sigmoid(binary_logits)
binary_prediction = (binary_probabilities >= 0.5).long()
```

## 7. 损失函数把任务变成优化目标

模型输出本身并不会告诉参数该怎么改。损失函数 `L(y_hat, y)` 把预测与目标之间的差异变成可微目标，训练希望降低训练样本上的平均损失。

以均方误差为例：

$
L=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
$

损失不是准确率；不同损失的数值尺度通常不能直接比较。对分类任务，准确率、precision、recall 等指标需要在模型输出上单独计算。训练损失下降也不保证验证集表现改善。

## 8. 反向传播：链式法则计算梯度

反向传播（backpropagation）是利用微积分链式法则，从损失开始沿计算图反向求出各参数对损失的偏导数。优化器随后使用这些梯度更新参数。反向传播负责“算梯度”，不是“更新参数”。

对一层仿射变换 `z = W a + b`，若已知上游梯度 `delta = dL/dz`，则：

$
\frac{\partial L}{\partial W}=\delta a^T,\qquad
\frac{\partial L}{\partial b}=\delta,\qquad
\frac{\partial L}{\partial a}=W^T\delta
$

梯度继续传回前一层；经过激活函数时，再乘以该激活函数的导数。对 ReLU，正输入处导数为 1，负输入处为 0。因此某个神经元若长期处在负区间，可能很少获得有效梯度（“死亡 ReLU”现象）。

在 PyTorch 中，`loss.backward()` 会从标量损失反向计算，并把梯度累积到叶子参数的 `.grad`：

```python
model = nn.Linear(2, 1)
x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
target = torch.tensor([[1.0], [0.0]])
loss = nn.MSELoss()(model(x), target)
loss.backward()

print(model.weight.grad.shape)  # [1, 2]
print(model.bias.grad.shape)    # [1]
```

默认情况下，梯度会累加而不是覆盖，所以常规训练每次更新前都要清空梯度。

## 9. 梯度下降与参数更新

最基础的梯度下降更新为：

$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L
$

`theta` 代表模型参数，`eta` 是学习率。学习率太大可能越过低损失区域甚至发散；太小则学习缓慢。实际中通常用 SGD、Adam 或 AdamW 等优化器管理参数更新。

```python
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for x, target in train_loader:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    loss = criterion(prediction, target)
    loss.backward()
    optimizer.step()
```

每一步的顺序是：清梯度、前向、计算损失、反向、更新参数。若忘了清梯度，连续 batch 的梯度会叠加；若忘了 `optimizer.step()`，参数不会更新。

## 10. 激活函数：为网络引入非线性

### 10.1 ReLU

$
\operatorname{ReLU}(x)=\max(0,x)
$

```python
nn.ReLU()
```

优点是计算简单、正区间梯度不随输入变小，常作为 MLP 隐藏层默认选择。负区间梯度为 0，若神经元长期只输出 0，可能难以恢复。较大学习率、不合适的偏置或数据分布都可能加重这种情况。

### 10.2 LeakyReLU 与 PReLU

LeakyReLU 在负区间保留一个较小斜率：

```python
nn.LeakyReLU(negative_slope=0.01)
```

它可以缓和零梯度区域；PReLU 则学习负区间斜率。它们并非总优于 ReLU，应结合验证表现选择。

### 10.3 Sigmoid

$
\sigma(x)=\frac{1}{1+e^{-x}}
$

输出在 `(0, 1)`，可解释为二分类概率（在模型校准等假设下）。当输入绝对值很大时曲线饱和、导数接近零，深层网络中连续使用容易造成梯度消失。现代隐藏层通常少用 sigmoid；二分类概率更常在推理时由 logits 经 sigmoid 得到。

### 10.4 Tanh

输出在 `(-1, 1)`，以 0 为中心，常用于早期网络或特定循环结构。但输入绝对值较大时也会饱和，导数趋近零，深层网络训练可能困难。

### 10.5 GELU 与 SiLU

GELU 和 SiLU（Swish）是平滑激活，在 Transformer 和现代网络中常见。它们通常比 ReLU 计算略复杂；选择激活函数要考虑模型结构、算力和经验基线。

### 10.6 输出层不应盲目加激活

回归通常让最后一层输出不受限实数；多分类输出 logits；二分类输出 logit。只有目标范围有明确约束时，才考虑输出 sigmoid、tanh 或其他变换，并应确保与损失定义不重复。

## 11. 梯度消失

反向传播需要沿网络层层相乘局部导数和权重矩阵。若大量因子绝对值小于 1，梯度的范数可能随着层数增加而快速衰减，靠近输入的早期层就几乎得不到更新，这称为梯度消失。

常见成因包括：

- 深层网络中连续使用饱和区的 sigmoid 或 tanh；
- 初始化尺度使信号/梯度逐层缩小；
- 激活长期落在 ReLU 的负半轴；
- 归一化、残差结构或学习率配置不适合当前网络。

典型表现是前层参数梯度范数很小、训练损失下降极慢，或网络加深后反而更难优化。可逐层检查激活均值/方差和梯度范数，而不是只看最终 loss。

常见缓解思路：使用 ReLU/GELU 等非饱和区性质较好的激活；采用 Xavier 或 Kaiming 等匹配激活的初始化；使用合适的归一化和残差连接；检查输入缩放与学习率。缓解措施要由证据驱动，不能看到训练慢就直接堆更复杂的模块。

## 12. 梯度爆炸

若链式相乘的因子整体大于 1，梯度范数可能层层放大，出现梯度爆炸。表现包括损失突然变成极大值或 `NaN`、参数更新剧烈、梯度范数远大于正常范围。

可以记录梯度范数：

```python
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        print(name, parameter.grad.norm().item())
```

梯度裁剪可以限制单次更新所依据的梯度范数：

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

梯度裁剪常用于循环网络或不稳定训练，但它不能代替修复错误输入、过大学习率或数值溢出。若梯度经常被强烈裁剪，应继续查根因。

## 13. 参数初始化

训练开始时尚无学习到的参数。初始化需要避免所有神经元完全相同（否则它们会产生相同梯度并学成相同功能），同时要让信号和梯度在网络传播时尺度不过度放大或缩小。

### 13.1 Xavier / Glorot 初始化

Xavier 初始化根据输入和输出连接数的 fan-in/fan-out 控制权重方差，常用于 tanh、线性层等：

```python
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)
```

### 13.2 Kaiming / He 初始化

Kaiming 初始化针对 ReLU 家族的非线性设计：

```python
nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
nn.init.zeros_(layer.bias)
```

### 13.3 在模型中应用初始化

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model.apply(init_weights)
```

`model.apply()` 会递归访问模块。输出层若不接 ReLU，可以使用不同初始化策略；示例展示统一基线，不代表所有层都应使用相同规则。自定义初始化应在创建模型后、训练前执行，加载预训练参数后不要再覆盖它们。

## 14. 输入标准化与数值尺度

MLP 对输入尺度常较敏感。若某一特征范围是 0–1，另一个是 0–100000，未缩放时大尺度特征可能主导线性组合和梯度。常见标准化为：

$
x'_{j}=\frac{x_j-\mu_j}{\sigma_j}
$

均值和标准差只能用训练集估计，再用于验证集、测试集和推理数据，避免数据泄漏。若某列标准差为零，需要单独处理。分类变量不一定适合当作连续数值做 z-score；可考虑 one-hot 或 embedding，视类别含义和任务而定。

目标值也可按需要缩放，尤其回归任务中目标尺度会影响损失和梯度。若标准化过目标，预测后应使用训练集对应参数反变换回业务单位。

## 15. 过拟合与泛化

模型在训练样本上表现很好，却在未参与训练的数据上表现较差，称为过拟合。MLP 参数很多时可能记住训练数据中的偶然细节。训练损失和验证损失常能提供线索：训练损失持续下降、验证损失先降后升，说明出现过拟合倾向。

反过来，训练和验证损失都很高且停滞，可能是欠拟合、输入或标签错误、优化没有生效，也可能是学习率不合适。应结合基线和数据检查，不应只凭一条曲线下结论。

## 16. 正则化方法

### 16.1 L2 正则化与 weight decay

在目标中加入权重平方惩罚：

$
L_{total}=L_{data}+\lambda\sum_j w_j^2
$

它倾向于限制权重过大。PyTorch 优化器可配置 `weight_decay`：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

AdamW 使用解耦权重衰减，是 Adam 类优化器的常用选择。不同 optimizer 对 `weight_decay` 的处理有差异；Bias 和归一化参数是否衰减也是常见参数组设计，不是统一硬性规则。

### 16.2 L1 正则化

L1 惩罚加入权重绝对值和：

$
L_{total}=L_{data}+\lambda\sum_j|w_j|
$

它可能推动部分权重变为零，产生稀疏参数。PyTorch 通常需要显式把惩罚加入 loss：

```python
l1_penalty = sum(parameter.abs().sum() for name, parameter in model.named_parameters()
                 if "weight" in name)
loss = task_loss + 1e-5 * l1_penalty
```

正则化强度 `lambda` 需要用验证集选择；太强会造成欠拟合。

### 16.3 Dropout

训练时随机屏蔽部分激活，降低神经元过度共适应：

```python
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(64, 2),
)
```

Dropout 仅在 `model.train()` 下生效，`model.eval()` 会关闭随机屏蔽并使用相应的期望缩放行为。不要在模型外手动按同样比例缩放激活。Dropout 对不同数据、网络和归一化组合效果不同。

### 16.4 Early stopping

每轮在验证集计算指标，若验证损失长时间没有改善则停止，并恢复验证指标最佳时的参数。Early stopping 限制继续拟合训练集的时间，但验证集被反复用于选择 epoch，因此最终测试集仍应独立保留。

### 16.5 数据层面的正则化

增加高质量训练样本、合理的数据增强或噪声建模，常能改善泛化。变换必须保持目标语义。数据增强不是对所有表格特征都适用，更不能取代按主体/时间/组正确划分数据集。

## 17. 归一化层与 MLP

BatchNorm 和 LayerNorm 都会改变激活的数值分布，但归一化维度和行为不同：

| 模块 | 主要归一化范围 | 常见特点 |
| --- | --- | --- |
| `BatchNorm1d` | batch 维（以及适用输入的其他维度），按特征统计 | 训练/评估行为不同，依赖 batch 统计 |
| `LayerNorm` | 每个样本的指定特征维 | 不依赖 batch 组成，Transformer/MLP 中常见 |

是否使用归一化层要结合网络深度、batch 大小和数据。标准化原始输入与网络内部的 BatchNorm/LayerNorm 是不同处理，不可混为一谈。归一化不是正则化的同义词，虽然某些情况下也会影响泛化。

## 18. 一个完整的 MLP 分类示例

以下示例假设 `train_loader`、`valid_loader` 每批返回 `features: [N, D]` 与 `labels: [N]` 的整数类别。输入特征应已用训练集统计量预处理。

```python
class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)  # logits

model = ClassifierMLP(input_dim=20, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

for epoch in range(50):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_count = 0

    for features, labels in train_loader:
        features = features.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        count = labels.size(0)
        train_loss_sum += loss.item() * count
        train_correct += (logits.argmax(dim=1) == labels).sum().item()
        train_count += count

    model.eval()
    valid_loss_sum = 0.0
    valid_correct = 0
    valid_count = 0
    with torch.inference_mode():
        for features, labels in valid_loader:
            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            logits = model(features)
            loss = criterion(logits, labels)

            count = labels.size(0)
            valid_loss_sum += loss.item() * count
            valid_correct += (logits.argmax(dim=1) == labels).sum().item()
            valid_count += count

    print(
        f"epoch={epoch + 1:03d} "
        f"train_loss={train_loss_sum / train_count:.4f} "
        f"train_acc={train_correct / train_count:.3f} "
        f"valid_loss={valid_loss_sum / valid_count:.4f} "
        f"valid_acc={valid_correct / valid_count:.3f}"
    )
```

训练 loss 按样本数加权后求平均，可正确处理最后一个较小 batch。验证时预测和正确数必须在每个 batch 内累计，分母使用整个验证集样本数。此示例未包含早停和保存最佳权重，实际使用时可按验证指标保存 checkpoint。

## 19. 回归 MLP 模板

回归的输出维度取决于目标数量，常用 `MSELoss` 或 `L1Loss`。预测和 target 的形状应保持一致，避免广播产生意外矩阵：

```python
model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
criterion = nn.MSELoss()

features = torch.randn(16, input_dim)
targets = torch.randn(16, 1)
predictions = model(features)
assert predictions.shape == targets.shape
loss = criterion(predictions, targets)
```

若模型输出 `[N, 1]`、标签却是 `[N]`，二者广播时可能生成 `[N, N]` 的比较，而不是逐样本对应。通过 `unsqueeze(1)` 或 `squeeze(1)` 明确统一 shape，并确认 dtype 为浮点数。

## 20. 训练诊断：先验证流程，再调模型

### 20.1 过拟合一个小 batch

在调复杂结构之前，可尝试反复训练同一个很小 batch。如果训练损失都不能明显下降，优先检查前向、标签、损失、梯度和 optimizer，而不是先加正则化：

```python
features, labels = next(iter(train_loader))
for step in range(200):
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(features.to(device)), labels.to(device))
    loss.backward()
    optimizer.step()
```

Dropout 和数据增强可能让这个检查不稳定；调试时可暂时简化网络/关闭随机操作，检查结束后恢复正式训练配置。

### 20.2 检查梯度是否存在且有限

```python
for name, parameter in model.named_parameters():
    if parameter.grad is None:
        print(name, "gradient is None")
    elif not torch.isfinite(parameter.grad).all():
        print(name, "gradient contains NaN/Inf")
    else:
        print(name, parameter.grad.norm().item())
```

参数梯度为 `None` 可能表示该参数没有参与当前计算图，或梯度还未计算；梯度全零也不一定是程序错误，但需要确认网络路径和激活行为。

### 20.3 常见表现与排查方向

| 表现 | 可能原因 | 检查方向 |
| --- | --- | --- |
| loss 不变化 | 未更新参数、梯度断开、标签/输出不匹配 | 确认 `backward()`、`step()`、参数梯度、shape |
| loss 变成 NaN | 学习率过高、输入有 NaN/Inf、数值溢出 | 检查输入、梯度、初始化和学习率 |
| 训练好而验证差 | 过拟合、数据拆分泄漏/分布差异 | 查重复样本、正则化和拆分策略 |
| 分类恒预测一类 | 类别不平衡、标签映射或阈值问题 | 查看类别比例、混淆矩阵、logits 和标签 |
| CUDA/MPS 设备错误 | 模型、输入或 target 在不同设备 | 检查 `.to(device)` 和 loss 输入 |
| 回归 loss 异常大 | 目标尺度、离群值、shape 广播 | 标准化目标、检查维度和单位 |

## 21. 参数量和模型容量

全连接层 `Linear(D_in, D_out)` 在带偏置时参数量为：

```text
D_in * D_out + D_out
```

连续的宽层会快速增加参数。例如从 1000 维输入映射到 512 维，单层已有超过 50 万个权重。参数越多不等于模型必然越好；小数据集上可能更容易过拟合，内存和计算成本也更高。选择隐藏宽度和层数时要同时看数据规模、验证结果和资源预算。

对于高分辨率图像，直接展平送入 MLP 会丢失局部空间归纳偏置并产生大量参数，CNN 往往更合适；对于表格向量、低维特征和固定长度嵌入，MLP 是常见基线。

## 22. 随机性与可复现性

初始化、DataLoader shuffle、Dropout 等都会引入随机性。可以设置种子以便比较实验：

```python
import random
import numpy as np

seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

相同种子不保证所有硬件和软件版本完全逐位一致；某些加速算子为性能会采用非确定性实现。数据拆分也需要单独固定随机种子。实验报告应记录环境、数据版本、拆分方式和随机种子。

## 23. MLP 学习自测

尝试先推理，再用 PyTorch 验证：

1. 两个 `Linear` 中间没有激活，为什么整体仍是线性/仿射函数？
2. 输入 `[N, 20]` 经过 `Linear(20, 64) -> ReLU -> Linear(64, 3)` 后，输出和分类标签分别是什么 shape/dtype？
3. 为什么 `CrossEntropyLoss` 前不应先做 softmax？`BCEWithLogitsLoss` 前为什么不应先做 sigmoid？
4. 深层 sigmoid 网络中，梯度为什么可能逐层变小？
5. Xavier 与 Kaiming 初始化分别通常和什么激活场景搭配？
6. 训练损失下降而验证损失上升，和两个损失都不下降，诊断方向有什么不同？
7. 为什么标准化参数只能从训练集估计？
8. `model.eval()` 是否会关闭 autograd？`torch.no_grad()` 是否会关闭 Dropout？

建议通过打印每层 shape、检查 `.grad`、绘制训练/验证损失曲线、在小 batch 上验证可学习性来核对答案。
