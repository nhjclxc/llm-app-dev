# PyTorch `torch.nn` 模块：从单层到完整网络

本文按“单层 → 多层 → 常用网络组件 → 模型参数与模式管理 → 完整练习”的顺序，系统学习 `torch.nn`。重点不是背 API，而是理解每层接收什么形状、输出什么形状、参数如何注册，以及各组件怎样组合成可训练模型。

示例基于 PyTorch 2.x，默认执行：

```python
import torch
from torch import nn
```

`torch.nn` 提供网络模块、层、激活、损失函数和参数管理机制。它定义模型及其前向计算；autograd 负责求梯度；`torch.optim` 根据梯度更新参数。损失函数和优化器在 [PyTorch02_nn和optimizer.md](./PyTorch02_nn和optimizer.md) 中另有专门介绍，本文会在模型例子中使用它们。

## 1. 学习地图和必要预备

| 学习阶段 | 主要内容 | 本文对应章节 |
| --- | --- | --- |
| 单层 | `nn.Module`、`nn.Linear`、激活函数 | 2–4 |
| 多层 | `nn.Sequential`、自定义 `forward()`、模块组合 | 5 |
| 常用网络组件 | Dropout、BatchNorm、LayerNorm、Embedding、卷积和池化 | 6–9 |
| 管理模型 | 参数、buffer、`state_dict()`、设备和模式切换 | 10–11 |
| 实际应用 | MLP 分类、文本嵌入练习、排错 | 12–14 |

在学习层之前，先熟悉张量形状。常见记号：`N` 表示批次大小，`D` 表示特征数，`C` 表示类别或通道数，`L` 表示序列长度，`V` 表示词表大小。`nn` 层通常不会自动猜测维度语义，输入形状必须符合该层约定。

## 2. 网络的基本单位：`nn.Module`

PyTorch 中的模型和绝大多数层都是 `nn.Module` 的子类。自定义模型时通常继承 `nn.Module`：

```python
class AffineModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

model = AffineModel(4, 2)
x = torch.randn(3, 4)
y = model(x)
print(y.shape)  # torch.Size([3, 2])
```

### 2.1 `__init__`、`forward()` 与 `model(x)`

- `__init__()`：创建并注册子模块、参数和 buffer。记得调用 `super().__init__()`。
- `forward(x)`：描述一次前向计算，即输入如何经过各层得到输出。
- `model(x)`：推荐调用方式。它会经过 `nn.Module` 的调用逻辑，包括 hooks 等机制，再进入 `forward()`。

一般不要直接写 `model.forward(x)`，否则会绕过模块调用入口上处理的一些机制。`forward()` 中只描述数据计算，不要在每次调用时重复创建本应作为模型状态保存的层：

```python
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)  # 注册一次，参数持续存在

    def forward(self, x):
        return self.fc(x)
```

若在 `forward()` 内部临时新建 `nn.Linear(...)`，其参数通常不属于模型稳定状态，也不会按预期被优化器持续更新。

### 2.2 `nn.Module` 自动管理什么

注册为子模块或参数后，`nn.Module` 可以递归地：

- 通过 `parameters()` 交给优化器；
- 通过 `.to(device)`、`.cuda()`、`.cpu()` 迁移参数和 buffer；
- 通过 `state_dict()` 保存状态；
- 通过 `train()` / `eval()` 切换整棵模块树的模式。

因此，层应放在 `self.layer`、`nn.Sequential`、`nn.ModuleList` 或 `nn.ModuleDict` 等受管理的结构中，而不是只放在普通 Python 容器里。

## 3. 单层入门：`nn.Linear`

### 3.1 线性层的计算

`nn.Linear(in_features, out_features)` 对输入最后一维做仿射变换：

\[
y = xW^T + b
\]

权重 `W` 的形状为 `[out_features, in_features]`，偏置 `b` 的形状为 `[out_features]`。默认 `bias=True`。

```python
layer = nn.Linear(in_features=4, out_features=3)
x = torch.randn(2, 4)
y = layer(x)

print(layer.weight.shape)  # [3, 4]
print(layer.bias.shape)    # [3]
print(y.shape)             # [2, 3]
```

批次维度不需要逐个循环处理：`Linear` 会对最后一维变换，并保留所有前置维度：

```python
layer = nn.Linear(8, 5)
x = torch.randn(2, 7, 8)  # 2 个样本，每个样本 7 个位置，每个位置 8 个特征
y = layer(x)
print(y.shape)  # [2, 7, 5]
```

如果输入为 `[N, D]`，则 `in_features` 必须等于 `D`。如果出现 `mat1 and mat2 shapes cannot be multiplied`，先打印输入 `shape` 和层的 `in_features`，通常是最后一维不匹配。

### 3.2 关闭偏置和读取权重

```python
no_bias = nn.Linear(4, 3, bias=False)
print(no_bias.weight.shape)  # [3, 4]
print(no_bias.bias)          # None
```

`weight` 和 `bias` 都是注册的可训练参数。实际项目中通常不直接手工改它们，而由优化器更新。若必须检查参数值，可在 `torch.no_grad()` 中读取或修改；打印参数可用 `layer.weight.detach()`。

### 3.3 单个样本与批次维度

对于 `nn.Linear(4, 2)`，输入 `[4]` 会得到 `[2]`；输入 `[N, 4]` 会得到 `[N, 2]`。数据加载器一般提供批次形式 `[N, D]`，保持批次维度能让训练和推理代码更统一。

## 4. 激活函数：给网络加入非线性

若多个线性层之间没有非线性，线性层的组合仍等价于一个线性变换，网络深度并不会带来更复杂的函数表达能力。激活函数使网络可以表示非线性关系。

### 4.1 `nn.ReLU`

$
\mathrm{ReLU}(x)=\max(0,x)
$

```python
relu = nn.ReLU()
x = torch.tensor([-2.0, 0.0, 3.0])
print(relu(x))  # tensor([0., 0., 3.])
```

ReLU 常用于 MLP、CNN 的隐藏层。默认 `inplace=False`；初学时优先保持非原地操作，以减少与 autograd 原地修改规则相关的问题。

### 4.2 `nn.GELU`

GELU 是平滑的门控型激活，常见于 Transformer：

```python
gelu = nn.GELU()
x = torch.tensor([-2.0, 0.0, 2.0])
print(gelu(x))  # 负值不会像 ReLU 一样全部变成 0
```

它不需要设置输入/输出维度，逐元素作用于输入。

### 4.3 `nn.Sigmoid`

Sigmoid 将数值映射到 `(0, 1)`：

```python
sigmoid = nn.Sigmoid()
probability = sigmoid(torch.tensor([-2.0, 0.0, 2.0]))
```

常用于需要概率形式的二分类输出或门控结构。但若训练时使用 `nn.BCEWithLogitsLoss`，应将未经过 sigmoid 的 logits 直接传给损失函数，由它执行数值稳定的组合计算；推理时如需概率，再调用 `torch.sigmoid(logits)`。

### 4.4 `nn.Softmax`

Softmax 沿指定维度把一组分数转换为和为 1 的概率：

```python
softmax = nn.Softmax(dim=1)
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.2, 0.3, 2.0]])
probabilities = softmax(logits)
print(probabilities.shape)             # [2, 3]
print(probabilities.sum(dim=1))        # 每个样本一行，结果约为 [1., 1.]
```

`dim` 非常重要：对 `[N, C]` 分类 logits，通常沿 `dim=1`（类别维）归一化；对 `[N, L, C]` 则通常用 `dim=-1`。如果沿批次维 softmax，得到的就不是每个样本的类别概率。

多分类训练一般使用 `nn.CrossEntropyLoss`，直接传入 logits，不要先套 `Softmax`。需要展示概率或按类别采样时再做 softmax。常用的 `argmax` 预测类别也不需要先做 softmax，因为 softmax 不改变最大值所在位置。

### 4.5 激活层的常见排列

常见 MLP 隐藏层顺序为“线性变换 → 激活”：

```python
block = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
)
```

最后一层是否使用激活取决于任务。回归通常输出原始值；多分类通常输出 logits；二分类搭配 `BCEWithLogitsLoss` 时也输出原始 logits。

logits 指的是模型最后一层输出的、尚未经过 Sigmoid 或 Softmax 等概率转换的原始分数。 你可以先把它理解成：模型对每个类别的“打分”，分数越高，模型通常越倾向于预测该类别。

注意 logits 就是分数，并不是概率，logits 输出的数值是会大于1的，要想转化为概率那么必须使用Softmax将其转化

## 5. 多层网络：`nn.Sequential` 与自定义 `forward()`

### 5.1 `nn.Sequential`

当数据只需按固定顺序依次经过各层，可用 `nn.Sequential` 快速搭建：

```python
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
)

x = torch.randn(5, 4)
logits = model(x)
print(logits.shape)  # [5, 3]
```

数据依次流过每个子模块。`Sequential` 会注册其中的模块和参数，适合线性堆叠结构。

带名称时可以使用 `OrderedDict`，方便在打印模型或读取状态字典时识别层：

```python
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ("hidden", nn.Linear(4, 16)),
    ("activation", nn.ReLU()),
    ("classifier", nn.Linear(16, 3)),
]))
```

### 5.2 自定义多层 MLP

当需要分支、跳连、多输入、条件逻辑或中间输出时，使用自定义 `forward()` 更灵活：

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        return self.classifier(x)  # 返回 logits

model = MLP(input_dim=784, hidden_dim=256, num_classes=10)
images = torch.randn(32, 784)
logits = model(images)
print(logits.shape)  # [32, 10]
```

### 5.3 多层形状追踪

以 `[N, 784] → [N, 256] → [N, 128] → [N, 10]` 为例：

```text
输入             [N, 784]
Linear(784, 256) [N, 256]
ReLU             [N, 256]
Linear(256, 128) [N, 128]
GELU             [N, 128]
Linear(128, 10)  [N, 10]
```

激活层通常不改变形状。逐层追踪形状是调试网络最有效的方法之一。

### 5.4 重复结构：`ModuleList` 和 `ModuleDict`

动态循环调用一组层时用 `nn.ModuleList`，而不是普通 `list`：

```python
class RepeatedMLP(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth)]
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
```

`ModuleList` 负责注册模块，但不会像 `Sequential` 那样自动定义前向顺序，需要自行在 `forward()` 中迭代。多个命名分支可用 `nn.ModuleDict`：

```python
class MultiTaskHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.heads = nn.ModuleDict({
            "classify": nn.Linear(hidden_dim, 4),
            "regress": nn.Linear(hidden_dim, 1),
        })

    def forward(self, features, task):
        return self.heads[task](features)
```

## 6. 正则化组件：`nn.Dropout`

Dropout 在训练时按概率随机将一部分激活置零，并对保留值进行缩放。它可减少神经网络对个别特征的过度依赖。

```python
dropout = nn.Dropout(p=0.5)
x = torch.ones(4, 8)

dropout.train()
y_train = dropout(x)   # 训练模式会随机置零并缩放

dropout.eval()
y_eval = dropout(x)    # 评估模式下等同于恒等映射
```

`p` 是每个元素被置零的概率，范围为 `[0, 1)`。在 MLP 中可将 Dropout 放在隐藏表示之后：

```python
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(64, 10),
)
```

注意：Dropout 的随机行为由模块的训练/评估模式控制。因此验证前必须调用 `model.eval()`，否则验证输出会随机波动。

## 7. 归一化组件

归一化层根据指定维度上的统计量调整激活，有助于改善优化过程。不同归一化对哪些维度计算均值和方差并不相同。

### 7.1 `nn.BatchNorm1d`

`BatchNorm1d(num_features)` 常用于全连接特征 `[N, C]`，也支持形状 `[N, C, L]` 的一维序列特征。它对每个通道/特征维使用批次统计量进行归一化：

```python
bn = nn.BatchNorm1d(num_features=32)
x = torch.randn(16, 32)  # N=16, C=32
y = bn(x)
print(y.shape)  # [16, 32]
```

BatchNorm 在训练模式下使用当前批次统计量，并更新运行均值和方差；评估模式下默认使用训练期间累积的运行统计量。运行统计量是 buffer，不是优化器更新的可训练参数。

对于 `[N, C, L]`，`num_features` 对应第二维 `C`：

```python
bn = nn.BatchNorm1d(num_features=16)
x = torch.randn(8, 16, 100)
y = bn(x)  # [8, 16, 100]
```

如果 batch 很小或训练/推理数据分布差异较大，BatchNorm 统计量可能不稳定。不要把 `[N, D]` 的 `D` 错当成批次维。

### 7.2 `nn.LayerNorm`

LayerNorm 对每个样本内部指定的最后若干维计算统计量，不依赖其他样本，因此小批次和变长序列场景也较稳定。`normalized_shape` 指定要归一化的尾部形状：

```python
layer_norm = nn.LayerNorm(64)
x = torch.randn(8, 20, 64)  # [N, L, D]
y = layer_norm(x)          # 对每个 token 的 D=64 个特征归一化
print(y.shape)             # [8, 20, 64]
```

如果想对每个样本的 `[L, D]` 整体归一化，可写 `nn.LayerNorm((20, 64))`；若序列长度变化，通常只归一化固定的特征维 `D`。LayerNorm 有可学习的缩放和偏移参数（默认 `elementwise_affine=True`），但没有 BatchNorm 那样依赖训练/评估阶段的运行统计量。

### 7.3 BatchNorm 与 LayerNorm 对照

| 特性 | `BatchNorm1d(C)` | `LayerNorm(D)` |
| --- | --- | --- |
| 常见输入 | `[N, C]` 或 `[N, C, L]` | `[..., D]` |
| 统计量范围 | 同一特征/通道的批次样本（及支持的序列位置） | 单个样本的指定特征维 |
| 依赖批次组成 | 是 | 否 |
| 训练/评估统计行为不同 | 是，训练批次统计量；评估运行统计量 | 通常相同，不维护运行均值/方差 |
| 常见场景 | MLP、CNN | Transformer、序列和小批量场景 |

MLP 中一个常见 block 是 `Linear → BatchNorm1d → ReLU`，也有根据任务采用不同顺序的结构；LayerNorm 常置于 Transformer 子层前后。层的顺序应与网络设计匹配。

## 8. 嵌入层：`nn.Embedding`

Embedding 是一个可训练的查找表，将离散整数索引映射到稠密向量。若词表大小为 `V`、嵌入维度为 `D`，其权重矩阵形状为 `[V, D]`。

```python
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)
token_ids = torch.tensor([[2, 8, 5], [9, 1, 0]])  # long 索引，形状 [N, L]
vectors = embedding(token_ids)

print(vectors.shape)  # [2, 3, 64]，即 [N, L, D]
```

索引值必须满足 `0 <= id < num_embeddings`，通常 dtype 为 `torch.long`。Embedding 权重会通过反向传播更新，被本次输入索引访问到的行会获得相应梯度。

### 8.1 Padding 索引

自然语言批处理常用 padding 填充不同长度的序列，可用 `padding_idx` 指定填充值索引：

```python
PAD_ID = 0
embedding = nn.Embedding(
    num_embeddings=10000,
    embedding_dim=128,
    padding_idx=PAD_ID,
)
token_ids = torch.tensor([[4, 7, 0, 0], [3, 8, 9, 0]])
vectors = embedding(token_ids)  # [2, 4, 128]
```

`padding_idx` 对应行不会按普通 Embedding 梯度方式更新，通常初始化为零。注意，Embedding 只负责索引到向量，不负责分词、位置编码、padding mask 或序列模型计算。

### 8.2 Embedding 与 one-hot 的关系

从数学上看，查找 embedding 等价于 one-hot 向量乘以 `[V, D]` 的嵌入矩阵，但实际使用 `nn.Embedding` 无需显式构造巨大的 one-hot 张量，内存和计算更高效。

## 9. 其他常见网络组件

### 9.1 展平层 `nn.Flatten`

图像输入通常是 `[N, C, H, W]`，进入全连接层前可保留批次维并展平其余维：

```python
flatten = nn.Flatten(start_dim=1)
images = torch.randn(8, 1, 28, 28)
features = flatten(images)
print(features.shape)  # [8, 784]
```

### 9.2 卷积层 `nn.Conv2d`

卷积层通过局部感受野提取空间特征，输入布局通常为 `[N, C_in, H, W]`：

```python
conv = nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
)
x = torch.randn(8, 1, 28, 28)
y = conv(x)
print(y.shape)  # [8, 16, 28, 28]
```

这里 `padding=1` 使 3×3 卷积在 stride 为 1 时保持高宽不变。二维卷积还有 `Conv1d`、`Conv3d` 对应一维和三维数据。

### 9.3 池化层

池化常用于降低空间分辨率：

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
features = pool(torch.randn(8, 16, 28, 28))
print(features.shape)  # [8, 16, 14, 14]
```

池化层通常没有可训练权重。自适应池化可将不同空间尺寸压到固定输出大小，例如 `nn.AdaptiveAvgPool2d((1, 1))`。

### 9.4 `nn.ModuleList`、`nn.ModuleDict` 与 `Sequential` 的选择

| 结构 | 自动注册子模块 | 自动按顺序执行 | 适用情形 |
| --- | --- | --- | --- |
| `nn.Sequential` | 是 | 是 | 固定的简单串行网络 |
| `nn.ModuleList` | 是 | 否 | 需要在循环或自定义逻辑中逐层调用 |
| `nn.ModuleDict` | 是 | 否 | 按名称选择多个层、分支或任务头 |
| Python `list` / `dict` | 否 | 否 | 不应直接用于需要模型管理的子模块 |

## 10. 参数管理：parameters、named_parameters 与 state_dict

### 10.1 `parameters()` 和 `named_parameters()`

`parameters()` 返回递归注册的参数，常传给优化器；`named_parameters()` 同时返回每个参数的路径名称：

```python
model = MLP(input_dim=784, hidden_dim=128, num_classes=10)

for parameter in model.parameters():
    print(parameter.shape)

for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.requires_grad)
```

被冻结的参数仍会出现在列表中，但 `requires_grad=False`；创建优化器时可筛选可训练参数：

```python
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_parameters, lr=1e-3)
```

参数数量可这样统计：

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total, trainable)
```

### 10.2 `state_dict()` 是什么

`state_dict()` 返回模型的可学习参数和持久化 buffer，键通常按模块路径命名：

```python
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)

for key, value in model.state_dict().items():
    print(key, value.shape)
```

它不包含模型 Python 类定义本身。加载权重时需要先创建结构相同的模型，再载入状态：

```python
state = model.state_dict()
new_model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
new_model.load_state_dict(state)
```

层名称和形状不匹配时，严格加载通常会报错。只有清楚理解缺失/多余键的含义时才考虑 `strict=False`。

保存和加载常见写法：

```python
torch.save(model.state_dict(), "model_weights.pth")

model = MLP(input_dim=784, hidden_dim=128, num_classes=10)
state = torch.load("model_weights.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state)
```

`weights_only` 参数取决于 PyTorch 版本；若版本不支持，应按照该版本文档处理。加载不可信来源的完整 pickle 文件存在安全风险。保存训练断点还可能需要 optimizer、epoch 和随机状态等，不只是模型权重。

### 10.3 Parameter 与 buffer

可训练状态用 `nn.Parameter`，不训练但希望随模型迁移、保存的张量可注册为 buffer：

```python
class Scale(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.register_buffer("running_count", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        return x * self.scale
```

`scale` 会出现在 `parameters()` 和 `state_dict()` 中；`running_count` 默认只在 `state_dict()` 中，不会交给优化器。BatchNorm 的运行均值和方差正是典型 buffer。

## 11. `train()`、`eval()` 与推理模式

```python
model.train()  # 训练模式
model.eval()   # 评估模式
```

这两个调用会设置模块及所有子模块的 `training` 状态，影响 Dropout、BatchNorm 等依赖模式的层：

- `train()`：Dropout 启用随机丢弃；BatchNorm 使用当前批次统计并更新运行统计。
- `eval()`：Dropout 关闭随机丢弃；BatchNorm 使用运行统计量。

`train()` / `eval()` **不负责开启或关闭 autograd**。推理或验证通常组合使用：

```python
model.eval()
with torch.inference_mode():
    logits = model(images)
```

如果验证阶段漏掉 `eval()`，Dropout/BatchNorm 仍按训练行为运行；如果推理阶段漏掉 `inference_mode()`，数值通常仍可计算，但会无谓记录梯度图、增加内存开销。

从验证返回训练时，应再次 `model.train()`。如果模型中有自定义模块，应让其遵守 `self.training` 状态约定。

## 12. 完整例子：用 MLP 分类 MNIST 张量

下面展示模型结构、输入形状、损失和更新过程。假设已经准备好形状为 `[N, 1, 28, 28]` 的图像以及 `[N]` 的整数类别标签：

```python
class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.network(images)  # logits: [N, 10]

model = MNISTMLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

images = torch.randn(32, 1, 28, 28)
labels = torch.randint(0, 10, (32,))

model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()

print(logits.shape)  # [32, 10]
print(loss.item())
```

交叉熵输入的是原始 logits，不要在 `forward()` 末尾加入 `Softmax`。需要预测类别时：

```python
model.eval()
with torch.inference_mode():
    logits = model(images)
    predicted_class = logits.argmax(dim=1)
    probabilities = torch.softmax(logits, dim=1)
```

### 12.1 验证阶段

```python
model.eval()
correct = 0
total = 0

with torch.inference_mode():
    for images, labels in validation_loader:
        logits = model(images)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
```

完整数据集训练还需要在 epoch 结束或训练结束后管理模型模式、设备和指标；数据读取和训练调度不属于 `torch.nn` 本身。

## 13. 文本输入例子：Embedding 接线性分类器

下面展示 token 索引如何经 Embedding 转为 `[N, L, D]`，再做简单池化并分类：

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, token_ids, mask):
        # token_ids / mask: [N, L]
        x = self.embedding(token_ids)  # [N, L, D]
        x = self.norm(x)               # [N, L, D]
        mask = mask.unsqueeze(-1)      # [N, L, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(x)      # [N, num_classes]

model = TextClassifier(vocab_size=10000, embedding_dim=128, num_classes=3)
token_ids = torch.tensor([[4, 7, 0], [3, 8, 9]], dtype=torch.long)
mask = token_ids.ne(0)
logits = model(token_ids, mask)
print(logits.shape)  # [2, 3]
```

这个例子是用于理解张量形状和模块组合的简化分类器，不含位置编码和注意力层。mask 用来避免 padding 位置影响平均池化；真实 NLP 任务还需根据 tokenizer、padding 约定和损失定义处理注意力掩码。

## 14. 常见错误与排查顺序

### 线性层维度错误

检查输入最后一维是否等于 `in_features`。图像接全连接层前，通常先 `Flatten(start_dim=1)`；要保留批次维，不能把所有元素都展平为一维。

### 分类准确率低或损失函数报错

- 单标签多分类通常用 logits `[N, C]` 和 `long` 标签 `[N]`，配合 `CrossEntropyLoss`。
- 二分类/多标签常用 logits 和浮点 0/1 标签，配合 `BCEWithLogitsLoss`。
- 训练损失时不要提前 Softmax/Sigmoid，除非所选损失明确定义需要概率。

### 参数没有被训练

检查层是否在 `__init__` 中创建并赋给 `self`，或放进 `ModuleList` / `ModuleDict` / `Sequential`。普通列表里的子模块不会自动注册。

### 验证结果不稳定

检查验证前是否 `model.eval()`。Dropout 会随机丢弃，BatchNorm 在训练模式下会使用当前批次统计量。

### 保存后无法加载

确保实例化的是结构兼容的模型，核对 `state_dict()` 键名及张量形状，并确认保存/加载的是权重还是完整对象。

### BatchNorm 报维度错误

核对输入布局。`BatchNorm1d(C)` 对二维输入要求 `[N, C]`；对于三维输入要求 `[N, C, L]`，`C` 必须等于 `num_features`。

### Embedding 索引越界或类型错误

确认 token id 是 `torch.long`，且所有值都在 `[0, num_embeddings)` 范围内；不要把 padding id 设置成词表范围之外的值。

## 15. 练习路线

1. 建立 `Linear(4, 8) → ReLU → Linear(8, 3)`，逐层打印输入输出形状。
2. 把 `Sequential` 改写为自定义 `nn.Module`，确认输出一致。
3. 依次替换 ReLU 为 GELU，并在隐藏层加入 Dropout；比较训练模式和评估模式的输出。
4. 在 MLP 中加入 BatchNorm1d，确认输入格式为 `[N, C]`；将它替换成 LayerNorm 并比较批次大小变化时的行为。
5. 运行 MNIST MLP，尝试调整隐藏维数、激活和 Dropout 概率，比较验证准确率。
6. 创建 Embedding，将一批 token id 映射为 `[N, L, D]`，再使用 LayerNorm 和线性层做简单分类。
7. 打印 `named_parameters()` 与 `state_dict()`，找出权重、偏置和 BatchNorm buffer 的对应名称。

## 16. 学习检查清单

- 能解释 `nn.Module`、`__init__`、`forward()` 和 `model(x)` 的职责。
- 能由 `Linear(in_features, out_features)` 推导权重形状和输出形状。
- 能解释为何线性层之间通常要有非线性激活。
- 知道 Softmax 的 `dim` 如何选择，以及为什么交叉熵训练通常接收 logits。
- 能区分 Sequential（自动串行执行）、ModuleList（注册但需自行执行）和普通 list。
- 能说明 Dropout 与 BatchNorm 如何受 `train()` / `eval()` 影响。
- 能根据输入维度选择 BatchNorm1d 或 LayerNorm，并正确设置特征维。
- 能说明 Embedding 的输入索引、权重表形状和输出 `[N, L, D]`。
- 能使用 `parameters()`、`named_parameters()`、`state_dict()` 检查与保存模型。
- 能在 MNIST MLP 中正确处理 `[N, 1, 28, 28]` 图像和 `[N, 10]` logits。

## 17. 推荐的后续学习

学完本章后，建议接着学习：

1. [PyTorch02_autograd.md](./PyTorch02_autograd.md)：理解 `backward()` 如何计算模型参数梯度。
2. [PyTorch02_nn和optimizer.md](./PyTorch02_nn和optimizer.md)：学习损失函数、优化器和完整训练循环。
3. 官方文档：[torch.nn](https://docs.pytorch.org/docs/stable/nn.html)、[PyTorch 构建神经网络](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)。

API 会随 PyTorch 版本迭代，查阅构造参数、支持的输入形状和设备行为时，应以安装版本对应的官方文档为准。
