# PyTorch 神经网络包 `nn` 和优化器 `optimizer` 相关教程

本文介绍 PyTorch 的神经网络包 `torch.nn` 和优化器包 `torch.optim`，按照“模块与层 -> 损失函数 -> 优化器 -> 训练流程 -> 工程技巧”逐步展开。示例默认已经执行：

```python
import torch
from torch import nn
```

## 1. `torch.nn` 和 `torch.optim` 的职责

典型训练流程由四部分构成：`nn.Module` 定义模型，损失函数衡量预测误差，autograd 计算梯度，optimizer 根据梯度更新参数。

```python
model.train()
for inputs, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

优化器不会自动计算损失，也不会读取数据；它只管理给定参数并根据参数的 `.grad` 执行更新。

## 2. `nn.Module` 基础

### 2.1 定义模型

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个参数表示输入特征是4个
        # 第二个参数表示输出特征是2个
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
x = torch.randn(3, 4)
y = model(x)  # 用 model(x)，不要直接调用 model.forward(x)
print(y.shape)  # [3, 2]
```

`__init__` 中定义子层，`forward` 中描述数据流。`super().__init__()` 用于初始化模块的参数和子模块注册机制。

### 2.2 参数注册

```python
model = MyModel()
print(list(model.named_parameters()))
print(model.state_dict().keys())
```

注册的参数才能由 `model.parameters()` 返回、随 `.to(device)` 迁移、被优化器更新并写入 `state_dict()`。层不能只放在普通 Python list 中；应使用 `ModuleList` 或 `Sequential`：

```python
class RepeatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### 2.3 训练模式和评估模式

```python
model.train()  # 训练模式
model.eval()   # 验证或推理模式

with torch.inference_mode():
    predictions = model(x)
```

`train()`/`eval()` 主要影响 Dropout、BatchNorm 等模块，并不等于开启/关闭梯度。推理时通常同时调用 `eval()` 和 `torch.inference_mode()`。

## 3. 常用网络层

### 3.1 全连接层 `nn.Linear`

`nn.Linear(in_features, out_features)` 对最后一维执行线性变换。输入最后一维必须等于 `in_features`：

```python
layer = nn.Linear(8, 3)
x = torch.randn(16, 8)
print(layer(x).shape)  # [16, 3]

# 可以带任意前置维度，Linear 只变换最后一维
x = torch.randn(2, 5, 8)
print(layer(x).shape)  # [2, 5, 3]
```

不需要偏置时可传入 `bias=False`。

### 3.2 卷积、池化和展平

二维图像卷积通常使用 `[N, C, H, W]` 布局：

```python
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
images = torch.randn(8, 3, 32, 32)
features = conv(images)
print(features.shape)  # [8, 16, 32, 32]

flatten = nn.Flatten(start_dim=1)  # 保留批次维度
print(flatten(features).shape)  # [8, 16384]
```

常见模块包括 `Conv1d/2d/3d`、`MaxPool2d`、`AvgPool2d` 和 `AdaptiveAvgPool2d`。卷积输出尺寸由 kernel、stride、padding、dilation 和输入尺寸共同决定，调试时逐层打印 shape。

### 3.3 循环网络

```python
lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=2, batch_first=True)
sequence = torch.randn(8, 20, 10)  # [batch, length, features]
output, (hidden, cell) = lstm(sequence)
print(output.shape)  # [8, 20, 32]
```

序列模块还包括 `nn.RNN`、`nn.GRU`。设置 `batch_first=True` 后使用 `[N, L, D]`；默认布局为 `[L, N, D]`。双向网络的输出特征通常为 `hidden_size * 2`。

## 4. 激活、正则化和归一化

### 4.1 激活函数

```python
relu = nn.ReLU()
gelu = nn.GELU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
leaky_relu = nn.LeakyReLU(0.01)
```

隐藏层常用 ReLU、GELU 或 SiLU。分类损失的输出约定见后文：通常将原始 logits 传给损失函数，不要提前 softmax/sigmoid。

### 4.2 Dropout 与归一化

```python
block = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.5))
bn = nn.BatchNorm1d(64)
bn_image = nn.BatchNorm2d(32)
ln = nn.LayerNorm(64)
```

Dropout 的 `p` 是训练时置零的概率，评估模式下自动关闭。BatchNorm 会维护运行统计量，训练和推理模式行为不同；LayerNorm 常沿最后一维归一化，常见于 Transformer。

### 4.3 权重初始化

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
model.apply(init_weights)
```

常见初始化有 Xavier、Kaiming 和正交初始化。初始化策略应结合层类型和激活函数选择，不要在训练中反复调用。

## 5. 模块容器、参数和 buffer

- `nn.Sequential`：固定顺序串接模块。
- `nn.ModuleList`：需要循环或条件控制地调用一组层。
- `nn.ModuleDict`：按名称组织多个分支或任务头。

```python
class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({
            "classify": nn.Linear(32, 3),
            "regress": nn.Linear(32, 1),
        })

    def forward(self, x, task):
        return self.heads[task](x)
```

可训练张量使用 `nn.Parameter`；不需要梯度、但需要随模型迁移和保存的张量注册为 buffer：

```python
class Scale(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        return x * self.weight
```

buffer 会进入 `state_dict()`，但不会出现在 `model.parameters()` 中。参数量统计和冻结参数：

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

for parameter in backbone.parameters():
    parameter.requires_grad_(False)
```

迁移学习中冻结主干后，应确认优化器仅包含可训练参数。后续解冻时，需重新创建优化器或通过 `add_param_group` 加入新参数。

## 6. 损失函数

损失函数将预测与目标转换为可优化的值。选择时要匹配输出形状、目标形状和 dtype。

| 任务 | 模型输出 | 目标 | 常用损失 |
| --- | --- | --- | --- |
| 回归 | 浮点预测 | 同形状浮点数 | `MSELoss`、`L1Loss`、`SmoothL1Loss` |
| 单标签多分类 | `[N, C]` 原始 logits | `[N]` 类别索引，`long` | `CrossEntropyLoss` |
| 二分类/多标签 | 原始 logits | 同形状 0/1 浮点值 | `BCEWithLogitsLoss` |

### 6.1 回归

```python
prediction = torch.tensor([[1.5], [3.0]])
target = torch.tensor([[1.0], [4.0]])
mse = nn.MSELoss()(prediction, target)
l1 = nn.L1Loss()(prediction, target)
smooth_l1 = nn.SmoothL1Loss()(prediction, target)
```

MSE 对大误差更敏感；L1 对异常值相对不敏感；SmoothL1 在两者之间折中。

### 6.2 多分类

```python
logits = torch.randn(8, 5)          # [N, C]，原始分数
target = torch.randint(0, 5, (8,))  # [N]，long 类别索引
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)
```

不要先对 logits 做 softmax；`CrossEntropyLoss` 内部已包含相应的数值稳定计算。类别不平衡时可以传类别权重，序列 padding 标签可用 `ignore_index` 忽略。

### 6.3 二分类和多标签

```python
logits = torch.randn(8, 1)
target = torch.randint(0, 2, (8, 1)).float()
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, target)

probability = torch.sigmoid(logits)  # 推理时需要概率再转换
prediction = (probability >= 0.5).long()
```

不要在传给 `BCEWithLogitsLoss` 前先 sigmoid。多标签任务的 logits 和 target 通常都是 `[N, C]`。

## 7. 优化器 `torch.optim`

### 7.1 SGD、Adam 与 AdamW

```python
sgd = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
)
adam = torch.optim.Adam(model.parameters(), lr=1e-3)
adamw = torch.optim.AdamW(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4
)
```

SGD 配合 momentum 简单稳定；Adam/AdamW 通常较容易在初期收敛。AdamW 将权重衰减与梯度更新解耦。学习率需结合 batch size、模型和数据尺度调试。

### 7.2 参数更新顺序

```python
optimizer.zero_grad(set_to_none=True)
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
```

`zero_grad` 清除上一步梯度，`backward` 计算新梯度，`step` 更新参数。遗漏 `zero_grad` 会导致梯度累积；把 `step()` 放在 `backward()` 前则不会使用当前损失的梯度。`set_to_none=True` 通常更节省内存。

### 7.3 参数组

不同模块可以使用不同学习率，例如迁移学习：

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-3},
], weight_decay=1e-4)
```

一个参数只能属于一个参数组。可通过 `optimizer.param_groups` 查看或修改当前学习率。

## 8. 学习率调度器

创建 scheduler 时传入 optimizer；应明确它是按 epoch 还是按训练 step 调整。

### 8.1 StepLR

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(num_epochs):
    train_one_epoch()
    scheduler.step()
```

### 8.2 CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-6
)
```

### 8.3 ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

for epoch in range(num_epochs):
    train_one_epoch()
    valid_loss = validate()
    scheduler.step(valid_loss)  # 需要传入监控指标
```

`ReduceLROnPlateau` 与普通 scheduler 不同，`step` 必须传入指标。可以用 `optimizer.param_groups[0]["lr"]` 记录当前学习率。

## 9. 完整 MLP 分类训练示例

下面示例串联模型、数据加载、损失、优化器、调度器和验证。这里使用随机标签，只用于演示流程，不代表真实任务效果。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)

features = torch.randn(1000, 20)
labels = torch.randint(0, 3, (1000,))
split = 800
train_loader = DataLoader(
    TensorDataset(features[:split], labels[:split]), batch_size=64, shuffle=True
)
valid_loader = DataLoader(
    TensorDataset(features[split:], labels[split:]), batch_size=128
)

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(10):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_count = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        count = batch_y.size(0)
        train_loss_sum += loss.item() * count
        train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        train_count += count

    scheduler.step()
    model.eval()
    valid_loss_sum = 0.0
    valid_correct = 0
    valid_count = 0
    with torch.inference_mode():
        for batch_x, batch_y in valid_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            count = batch_y.size(0)
            valid_loss_sum += loss.item() * count
            valid_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            valid_count += count

    print(
        f"epoch={epoch + 1:02d} "
        f"lr={optimizer.param_groups[0]['lr']:.6f} "
        f"train_loss={train_loss_sum / train_count:.4f} "
        f"train_acc={train_correct / train_count:.3f} "
        f"valid_loss={valid_loss_sum / valid_count:.4f} "
        f"valid_acc={valid_correct / valid_count:.3f}"
    )
```

损失按样本数加权，避免最后一个较小批次被赋予过大权重。验证时调用 `eval()` 并禁用梯度，不执行 `backward()` 或 `optimizer.step()`。

## 10. 梯度裁剪、累积和混合精度

### 10.1 梯度裁剪

在反向传播后、参数更新前裁剪梯度范数：

```python
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

裁剪能限制过大的梯度，但不能替代对数据、模型和学习率的排查。

### 10.2 梯度累积

显存不足时，可以把多个小批次的梯度累积后再更新：

```python
accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)

for step, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs.to(device))
    targets = targets.to(device)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

epoch 末若剩余批次不足一个累积周期，也需补做一次更新。损失除以累积步数，使总梯度尺度接近大批次平均梯度。

### 10.3 CUDA 自动混合精度

```python
amp_enabled = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
optimizer.zero_grad(set_to_none=True)

with torch.autocast(
    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
):
    outputs = model(inputs.to(device))
    loss = criterion(outputs, targets.to(device))

scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # 梯度裁剪前先反缩放
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

先确保普通精度训练正确，再尝试 AMP。硬件、PyTorch 版本和算子会影响支持情况与收益；出现 NaN 时可关闭 AMP 定位。

## 11. 保存模型和断点续训

### 11.1 保存模型权重

推荐保存 `state_dict`，而不是直接保存完整 Python 模型对象：

```python
torch.save(model.state_dict(), "model_weights.pt")

new_model = MLP().to(device)
state_dict = torch.load(
    "model_weights.pt", map_location=device, weights_only=True
)
new_model.load_state_dict(state_dict)
new_model.eval()
```

加载前需要创建相同结构的模型。只加载可信来源的 checkpoint。

### 11.2 保存训练状态

```python
checkpoint = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_valid_loss": best_valid_loss,
}
torch.save(checkpoint, "checkpoint.pt")
```

恢复时先创建相同模型、优化器和调度器，再加载状态：

```python
checkpoint = torch.load(
    "checkpoint.pt", map_location=device, weights_only=True
)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
start_epoch = checkpoint["epoch"] + 1
best_valid_loss = checkpoint["best_valid_loss"]
```

使用混合精度、随机采样器或其他有状态组件时，也应保存对应状态。

## 12. 常见问题排查

### 12.1 优化器没有更新参数

```python
for name, parameter in model.named_parameters():
    print(name, parameter.requires_grad, parameter.grad is None)
```

检查参数是否注册、是否 `requires_grad=True`、是否传入 optimizer、计算是否在 `no_grad` 中，以及是否执行 `backward()` 和 `step()`。

### 12.2 损失为 NaN 或 Inf

```python
print(torch.isfinite(inputs).all())
print(torch.isfinite(loss))
for parameter in model.parameters():
    if parameter.grad is not None:
        print(torch.isfinite(parameter.grad).all())
```

重点排查学习率过大、输入包含非有限值、损失和输出不匹配、除零、非法 log 以及混合精度溢出。

### 12.3 效果或设备异常

- 训练集指标改善而验证集变差：检查过拟合、数据划分和验证预处理。
- 小批数据也无法拟合：检查标签、输出维度、损失、梯度流和优化器参数。
- GPU 提示设备不一致：统一模型、输入、目标及相关 buffer 的 device。
- 验证结果异常：确保调用 `model.eval()`，并固定验证集。

| 任务 | 输出 | 损失 | 关键约定 |
| --- | --- | --- | --- |
| 多分类 | `[N, C]` 原始 logits | `CrossEntropyLoss` | 不先 softmax，标签为 long 索引 |
| 二分类 | `[N, 1]` 原始 logits | `BCEWithLogitsLoss` | 不先 sigmoid，目标为浮点 0/1 |
| 多标签 | `[N, C]` 原始 logits | `BCEWithLogitsLoss` | 目标与输出同形状 |
| 回归 | 浮点预测 | `MSELoss` 等 | 输出和目标形状匹配 |

## 13. 进阶：梯度 hook

```python
def show_gradient(gradient):
    print("gradient norm:", gradient.norm().item())

parameter = next(model.parameters())
handle = parameter.register_hook(show_gradient)
loss.backward()
handle.remove()
```

Hook 可用于观察梯度或模块中间值，但会增加复杂度和运行开销。调试后及时移除，避免训练循环里重复注册。

## 14. 推荐学习顺序

1. 用 `nn.Linear`、ReLU 和 `CrossEntropyLoss` 写二层 MLP。
2. 核对 logits、标签 shape、dtype 与损失函数要求。
3. 使用 `Dataset`/`DataLoader`，分离训练集和验证集。
4. 加入 Dropout、权重衰减和学习率调度，比较验证表现。
5. 保存最佳权重并恢复 checkpoint。
6. 学习卷积、归一化和池化层，逐层跟踪张量形状。
7. 基础训练稳定后再试 AMP、梯度累积、参数组和迁移学习。

最终应能解释每层的输入输出形状、损失函数需要的预测和标签格式，以及优化器实际更新了哪些参数。围绕这三点检查，可快速定位大部分 `nn` 与 optimizer 的初学问题。


超参数如何确定？激活函数如何选择？损失函数如何选择？优化器如何选择？