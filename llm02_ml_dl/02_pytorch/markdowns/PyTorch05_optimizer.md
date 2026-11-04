# PyTorch 优化器 `torch.optim`：从 0 到 1

本文从梯度下降开始，系统学习 PyTorch 优化器、常用算法、学习率调度、梯度裁剪、混合精度、断点恢复，以及不同任务下的选择方法。

示例默认执行：

```python
import torch
from torch import nn
```

## 1. 优化器在训练流程中的位置

```python
optimizer.zero_grad(set_to_none=True)
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
```

`nn.Module` 定义参数和前向计算，损失函数产生优化目标，autograd 计算参数 `.grad`，optimizer 读取梯度并更新参数，scheduler 修改 optimizer 的超参数。优化器不会自动计算损失，也不会自动调用 `backward()`。

## 2. 梯度下降基础

梯度下降的更新公式为：

$
\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)
$

`eta` 是学习率。梯度指向损失上升最快的方向，因此减去梯度可以降低局部损失。

```python
w = torch.tensor(5.0, requires_grad=True)
for _ in range(3):
    loss = (w - 2) ** 2
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
    w.grad = None
```

优化器封装了上述逻辑，并额外管理动量、指数移动平均和权重衰减等状态。

### 2.1 梯度为什么要清零

`.backward()` 默认累积梯度：

```python
x = torch.tensor(2.0, requires_grad=True)
(x ** 2).backward()
(3 * x).backward()
print(x.grad)  # 7，而不是 3
```

因此一次参数更新通常对应一次：

```python
optimizer.zero_grad(set_to_none=True)
```

`set_to_none=True` 通常减少内存写入；代码需要接受参数 `.grad` 可能是 `None`。

## 3. 第一个优化器

```python
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[3.0], [5.0], [7.0]])

for step in range(100):
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

`params` 是要管理的参数；`lr` 是学习率；`momentum` 控制 SGD 动量；`weight_decay` 控制权重衰减；Adam 类还会使用 `betas` 和 `eps`。

## 4. Optimizer 核心 API

```python
optimizer.zero_grad(set_to_none=True)  # 清空梯度
loss.backward()                        # 计算梯度
optimizer.step()                       # 更新参数
```

查看和修改参数组：

```python
for group in optimizer.param_groups:
    print(group["lr"], len(group["params"]))
optimizer.param_groups[0]["lr"] = 1e-4
```

优化器状态包含动量和自适应统计量：

```python
state = optimizer.state_dict()
print(state.keys())  # state, param_groups
```

只保存模型权重适合推理；要无缝续训，还要保存 optimizer 和 scheduler 状态。

## 5. 参数组与冻结参数

不同层可以使用不同学习率和权重衰减：

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-4},
    {"params": model.head.parameters(), "lr": 1e-3, "weight_decay": 0.0},
], weight_decay=1e-2)
```

组内值会覆盖默认值；一个参数不能重复属于同一个 optimizer 的多个组。

冻结主干时：

```python
for p in model.backbone.parameters():
    p.requires_grad_(False)
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad), lr=1e-3
)
```

后来解冻的参数需要重新创建 optimizer，或使用 `optimizer.add_param_group(...)` 加入。

## 6. SGD 与 Momentum

普通 SGD：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Momentum 累积历史方向，减少震荡：

```python
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, nesterov=True
)
```

SGD 简单、额外状态少，但对学习率敏感。图像分类中，SGD + momentum 配合较长训练和学习率调度常有很好的最终泛化表现。

## 7. Adam 与 AdamW

Adam 为每个参数保存梯度的一阶、二阶指数移动平均：

$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8
)
```

Adam 通常容易作为新任务的快速基线，但每个参数有额外状态，显存占用高于 SGD。

AdamW 将权重衰减与自适应梯度更新解耦：

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-2
)
```

Transformer、语言模型和现代视觉模型常以 AdamW 作为起点。经验上常对 bias 和归一化层不做衰减：

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if p.ndim == 1 or name.endswith(".bias") else decay).append(p)
optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 1e-2},
    {"params": no_decay, "weight_decay": 0.0},
], lr=1e-3)
```

## 8. 其他常见优化器

```python
rmsprop = torch.optim.RMSprop(model.parameters(), lr=1e-3)
adagrad = torch.optim.Adagrad(model.parameters(), lr=1e-2)
```

- RMSprop 按梯度平方移动平均缩放更新，历史上常见于循环网络。
- Adagrad 适合稀疏特征，但累计平方梯度可能使学习率过早变小。
- Adadelta 改进了 Adagrad，如今使用相对少。
- LBFGS 是准二阶方法，需要闭包重新计算 loss，适合小规模平滑问题。

## 9. 优化器如何选择

| 场景 | 推荐起点 | 注意点 |
| --- | --- | --- |
| 快速验证模型能否训练 | Adam 或 AdamW | 初始收敛通常快 |
| Transformer、语言模型 | AdamW | 常配 warmup 和余弦调度 |
| 图像分类追求最终泛化 | SGD + momentum | 通常要更长训练 |
| 稀疏特征 | Adagrad 等 | 观察学习率是否过早衰减 |
| 小型平滑、全批量问题 | LBFGS | 不适合常规大模型小批训练 |
| 不确定从哪里开始 | AdamW，`lr=1e-3` 左右 | 先建立基线 |

实用决策流程：

1. 先用 AdamW 建立可收敛基线。
2. 固定模型和数据，测试学习率的不同数量级。
3. 记录训练/验证损失、指标、学习率和梯度范数。
4. 若最终泛化需要改善，尝试 SGD + momentum。
5. 一次只改变一个主要因素。

优化器不能修复错误标签、数据泄漏、维度错位或错误损失。先确认模型能在很小的数据集上过拟合，再比较优化器。

## 10. 学习率与调度器

学习率太大时会震荡、发散或产生 NaN；太小时损失下降慢。可以先测试 `1e-5`、`1e-4`、`1e-3`、`1e-2` 等数量级，并记录每个参数组的当前学习率：

```python
for i, group in enumerate(optimizer.param_groups):
    print(i, group["lr"])
```

### 10.1 `StepLR`、`MultiStepLR`、`CosineAnnealingLR`

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)

for epoch in range(30):
    train_one_epoch(model, optimizer)
    scheduler.step()
```

`MultiStepLR` 在指定节点衰减；`CosineAnnealingLR` 按余弦曲线下降：

```python
multi = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)
```

通常先执行 `optimizer.step()` 再执行 `scheduler.step()`。`T_max` 与 scheduler 的调用次数对应；每 batch 调用就按 step 理解，每 epoch 调用就按 epoch 理解。

### 10.2 `ReduceLROnPlateau`

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
)
for epoch in range(20):
    train_loss = train_one_epoch(model, optimizer)
    valid_loss = validate(model)
    scheduler.step(valid_loss)  # 必须传入监控指标
```

监控 accuracy 等越大越好的指标时使用 `mode="max"`。它与 `StepLR` 不同，`step()` 必须接收指标。

### 10.3 OneCycleLR 与 LambdaLR

`OneCycleLR` 通常按每个 batch 更新，先升高再降低学习率；`LambdaLR` 用函数自定义倍率：

```python
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: 0.95 ** epoch
)
```

使用 scheduler 前先明确它按 epoch 还是 batch 更新，以及续训时是否恢复 scheduler 状态。

## 11. 梯度裁剪、累积和混合精度

### 11.1 梯度裁剪

梯度裁剪放在 `backward()` 后、`optimizer.step()` 前：

```python
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=1.0
)
optimizer.step()
```

`clip_grad_value_` 可限制每个梯度元素。裁剪不能修复 NaN；若裁剪前就非有限，应检查学习率、输入、损失和模型数值稳定性。

### 11.2 梯度累积

显存不足时用多个小 batch 模拟大 batch：

```python
accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)
for step, (inputs, targets) in enumerate(train_loader):
    loss = criterion(model(inputs), targets) / accumulation_steps
    loss.backward()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

将 loss 除以累积步数，使累积梯度近似大 batch 平均损失的梯度。最后不足一个周期的剩余梯度要额外处理；scheduler 也应按真实 optimizer 更新次数设计。

### 11.3 混合精度

在支持的设备上，autocast 配合 GradScaler：

```python
scaler = torch.amp.GradScaler("cuda")
for inputs, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(inputs), targets)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # 裁剪前必须反缩放
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

不同 PyTorch 版本对 AMP API 的推荐写法可能不同，以当前版本文档为准。

## 12. 完整训练循环

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_items = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n = inputs.size(0)
        total_loss += loss.detach().item() * n
        total_items += n
    return total_loss / total_items


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_items = 0.0, 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(model(inputs), targets)
            n = inputs.size(0)
            total_loss += loss.item() * n
            total_items += n
    return total_loss / total_items


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
for epoch in range(20):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    valid_loss = evaluate(model, valid_loader, criterion, device)
    scheduler.step()
    print(epoch, train_loss, valid_loss, optimizer.param_groups[0]["lr"])
```

这里用样本数加权累计 loss，避免最后一个小 batch 对 epoch 平均值产生过大影响。

## 13. 保存和恢复训练断点

```python
checkpoint = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_valid_loss": best_valid_loss,
}
torch.save(checkpoint, "checkpoint.pth")
```

恢复时先创建结构相同的模型、optimizer 和 scheduler，再载入状态：

```python
checkpoint = torch.load("checkpoint.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
start_epoch = checkpoint["epoch"] + 1
best_valid_loss = checkpoint["best_valid_loss"]
```

只做推理时保存模型权重即可；要无缝续训则必须保存 optimizer 和 scheduler 状态。`weights_only` 是否可用取决于 PyTorch 版本，加载不可信 pickle 文件存在安全风险。

## 14. 优化器没有更新参数时的排查

1. 参数是否注册：`list(model.named_parameters())` 是否包含目标层。
2. 参数是否传给 optimizer：检查 `optimizer.param_groups`。
3. 参数是否 `requires_grad=True`。
4. loss 是否被 `detach()`、`.item()` 或 `no_grad()` 断开。
5. 是否执行 `zero_grad`、`backward` 和 `step`。
6. `step()` 前梯度是否为 `None`、全零、NaN 或 Inf。
7. 学习率是否为 0，是否被 scheduler 降得过小。
8. 标签、输出形状和损失函数是否匹配。

```python
for name, parameter in model.named_parameters():
    grad = parameter.grad
    info = None if grad is None else {
        "norm": grad.norm().item(),
        "finite": torch.isfinite(grad).all().item(),
    }
    print(name, parameter.requires_grad, info)
```

| 症状 | 优先检查 |
| --- | --- |
| loss 完全不变 | 是否 `backward`/`step`，参数是否在 optimizer 中 |
| loss 为 NaN | 学习率、输入、梯度爆炸、混合精度溢出 |
| loss 降低但验证变差 | 过拟合、weight decay、数据增强、模式切换 |
| 训练前几步剧烈震荡 | 降低学习率，检查 loss reduction 和标签 |
| 显存持续增长 | 是否保存带图 tensor，是否错误保留计算图或状态 |
| 恢复后行为突变 | 是否同时恢复 optimizer/scheduler 状态 |

## 15. 实验设计与最终选择

比较优化器时保持数据切分、随机种子、模型初始化、epoch、batch size、评价指标和预处理一致。至少记录训练/验证 loss、验证指标、各参数组学习率、梯度范数和训练耗时。

推荐实验顺序：

1. AdamW，`lr=1e-3`，小权重衰减，作为可收敛基线。
2. 在相同设置下测试学习率的不同数量级。
3. 加入 scheduler，比较收敛速度和最终验证指标。
4. 在图像分类等适合任务上测试 SGD + momentum。
5. 最后再调 batch size、warmup 和 weight decay 等相互影响的参数。

不要只根据训练 loss 最低选择优化器；最终目标是验证/测试表现、稳定性、速度和显存的综合平衡。

## 16. 学习检查清单

- 能解释梯度、学习率和 `optimizer.step()` 的关系。
- 能写出 `zero_grad → forward → loss → backward → step`。
- 能区分 SGD、Momentum、Adam 和 AdamW 的更新思想。
- 能理解参数组、`param_groups` 和 optimizer `state_dict()`。
- 能按 epoch 或 batch 正确调用学习率 scheduler。
- 能在混合精度下先 `unscale_` 再做梯度裁剪。
- 能保存并恢复模型、optimizer 和 scheduler 的完整状态。
- 能根据收敛速度、泛化、显存和稳定性选择优化器。

## 17. 推荐学习顺序与资料

1. 手动实现第 2 节的单参数梯度下降。
2. 运行第 3 节线性回归，观察不同学习率现象。
3. 使用同一个模型比较 SGD、Adam、AdamW。
4. 加入 StepLR、CosineAnnealingLR 和 ReduceLROnPlateau。
5. 验证梯度裁剪、参数冻结和断点恢复。
6. 进行控制变量实验，形成自己的优化器选择经验。

- [PyTorch `torch.optim` 官方文档](https://docs.pytorch.org/docs/stable/optim.html)
- [PyTorch 优化循环教程](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [PyTorch 学习率调度器文档](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

具体构造参数、AMP 接口和 scheduler 行为可能随 PyTorch 版本变化，实际运行时应优先查阅与当前安装版本匹配的官方文档。
