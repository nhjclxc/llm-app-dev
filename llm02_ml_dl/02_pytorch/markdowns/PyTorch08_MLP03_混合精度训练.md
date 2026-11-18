# PyTorch 混合精度训练：从原理到完整实践

混合精度训练（Automatic Mixed Precision，AMP）是在同一次训练中，根据运算的数值特性选择不同浮点精度。通常让矩阵乘法、卷积等适合加速的运算使用 FP16 或 BF16，让参数更新、归一化、损失计算等敏感部分保留 FP32。

混合精度不是简单把整个模型改成 float16。可靠 AMP 通常由两部分组成：

1. autocast：前向计算时自动选择算子 dtype。
2. GradScaler：FP16 训练时放大 loss，减少梯度下溢。

本文从 FP32、FP16、BF16 的表示范围开始，逐步学习 autocast、GradScaler、完整训练循环、验证、梯度裁剪、梯度累积、checkpoint、性能测试和故障排查。代码以 PyTorch 2.x 为主，实际支持取决于当前版本、硬件和算子实现。

## 1. 为什么需要混合精度

训练中大量时间消耗在矩阵乘法和卷积。支持低精度加速的硬件可以用 FP16/BF16 降低显存和带宽压力：激活张量占用更少显存、同样显存可使用更大 batch、GPU 可能获得更高吞吐。

低精度同时带来更小的数值范围或更少的有效位，可能出现舍入、溢出、下溢、Inf 和 NaN。AMP 的目标是在性能和稳定性之间做选择。

```text
FP32 参数与优化器状态
        |
        +-- autocast：前向部分运算使用 FP16/BF16
        |
        +-- GradScaler：FP16 反向前放大 loss
        |
        +-- unscale：恢复真实梯度并检查非有限值
        |
        +-- optimizer.step：更新参数
```

```python
epoch = 1
train_loss = 0.82
valid_loss = 0.91
print(epoch, train_loss, valid_loss)
```

TensorBoard 或日志系统只记录数值，不能替代正确的训练流程和指标统计。

## 2. 浮点数的基本概念

浮点数由符号、指数和尾数组成。指数影响可表示范围，尾数影响有效精度。不同 dtype 的范围和相邻数间隔不同。

```python
import torch

for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
    info = torch.finfo(dtype)
    print(dtype)
    print("bytes:", torch.tensor([], dtype=dtype).element_size())
    print("max:", info.max)
    print("tiny:", info.tiny)
    print("eps:", info.eps)
```

eps 表示 1 附近的有效间隔量级，tiny 是接近 0 的正常正数范围。模型实际还受硬件行为和算子内部累积精度影响。

## 3. FP32、FP16、BF16

FP32 每个元素通常占 4 字节，范围和精度较平衡，是最稳妥的训练基线。

```python
x = torch.tensor([1.0, 2.0], dtype=torch.float32)
print(x.dtype, x.element_size())
```

FP16 每个元素占 2 字节，在支持的 GPU 上速度和显存表现好，但指数范围较小，较大激活容易溢出，较小梯度容易下溢。

```python
x = torch.tensor([1.0, 2.0], dtype=torch.float16)
print(x.dtype, x.element_size())
print(torch.finfo(torch.float16))
```

BF16 也占 2 字节，指数范围接近 FP32，通常更不容易溢出，但尾数精度低于 FP16。支持 BF16 的硬件上，BF16 常更稳定。

```python
for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    info = torch.finfo(dtype)
    print(dtype, "max=", info.max, "eps=", info.eps)
```

BF16 并非所有设备都支持。应先检查硬件和 PyTorch 版本，再做 benchmark。

## 4. 下溢、溢出与 NaN

梯度小到低精度无法表示时可能下溢为 0；激活或梯度过大时可能变成 Inf，后续计算可能产生 NaN。

```python
small = torch.tensor([1e-8], dtype=torch.float32)
large = torch.tensor([1e5], dtype=torch.float32)
print("FP32:", small, large)
print("FP16:", small.to(torch.float16), large.to(torch.float16))
```

具体打印结果可能因硬件和版本略有不同，重点是理解低精度的有限表示范围。

```python
value = torch.tensor([float("nan"), float("inf"), 1.0])
print(torch.isfinite(value))
print(torch.isfinite(value).all().item())
```

## 5. AMP 不等于 model.half()

不建议把整个模型和输入都手动改成 FP16：

```python
# 不推荐作为通用 AMP 方案
# model = model.half()
# features = features.half()
```

这种方式让参数、激活和所有算子都进入低精度，容易遇到归一化、损失、第三方算子和 dtype 不匹配问题。AMP 通常保留模型参数为 FP32：

```python
from torch import nn

model = nn.Linear(4, 2).float()
features = torch.randn(8, 4, dtype=torch.float32)
print(next(model.parameters()).dtype, features.dtype)
```

autocast 只影响上下文中的算子选择，不会把模型参数永久变成低精度。

## 6. 设备能力检查

CUDA GPU 是最常见的 AMP 训练设备。新版本 PyTorch 可能支持 CPU BF16 autocast；MPS 的支持和收益依版本、硬件与算子而异。

```python
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print("mps:", mps_available)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print("bf16:", torch.cuda.is_bf16_supported())
```

选择设备时不要写死 cuda：

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)
```

## 7. autocast 原理与使用

torch.autocast 是上下文管理器，根据设备、输入 dtype 和算子实现自动选择计算精度。一个上下文内并非所有运算都会变成同一种 dtype。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Linear(8, 4).to(device)
x = torch.randn(2, 8, device=device)
amp_enabled = device.type == "cuda"

with torch.autocast(
    device_type=device.type,
    dtype=torch.float16,
    enabled=amp_enabled,
):
    y = model(x)

print("parameter:", next(model.parameters()).dtype)
print("input:", x.dtype)
print("output:", y.dtype)
```

autocast 主要包住前向和 loss，不应包住 optimizer.step、数据加载和 checkpoint 保存。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
labels = torch.randint(0, 2, (2,), device=device)

optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
    logits = model(x)
    loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

## 8. dtype 配置

CUDA FP16 常用 torch.float16；CUDA BF16 需要硬件支持。CPU 常见 BF16，但是否有收益需测试。MPS 不应直接照搬 CUDA 配置。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp_enabled = True
else:
    amp_dtype = torch.float32
    amp_enabled = False
print(device, amp_dtype, amp_enabled)
```

用统一上下文：

```python
with torch.autocast(
    device_type=device.type,
    dtype=amp_dtype,
    enabled=amp_enabled,
):
    value = torch.randn(4, device=device).square().sum()
print(value)
```

初学者先验证 FP32 训练正确，再只改变 AMP 配置，比较速度、显存和指标。

## 9. GradScaler 原理

FP16 梯度可能下溢。GradScaler 在反向前把 loss 放大：

```text
scaled_loss = loss * scale
scaled_gradient = true_gradient * scale
真实梯度 = scaled_gradient / scale
```

如果放大后的梯度出现 Inf/NaN，scaler 会跳过本次参数更新并降低 scale；若多次正常则可能提高 scale。

```python
loss = torch.tensor(0.001)
scale = 1024.0
print("loss:", loss.item())
print("scaled:", (loss * scale).item())
```

BF16 动态范围接近 FP32，通常不需要 scaling，但实际行为需结合任务确认。

## 10. 创建 GradScaler

现代 PyTorch 2.x 常见接口是 torch.amp.GradScaler：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_enabled = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
print(scaler.is_enabled())
```

旧项目可能使用 torch.cuda.amp.GradScaler 和 torch.cuda.amp.autocast。新代码优先采用当前版本推荐接口，并核对版本差异。

```python
# 旧接口示意，仅用于识别旧代码
# scaler = torch.cuda.amp.GradScaler()
# with torch.cuda.amp.autocast():
#     output = model(x)
```

无 CUDA 时 enabled=False 会走普通 FP32 分支。

## 11. 最小 AMP 训练步骤

正确顺序：清零梯度、autocast 前向、计算 loss、scale 后 backward、可选 unscale/裁剪、scaler.step、scaler.update。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Linear(4, 2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
amp_enabled = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

features = torch.randn(8, 4, device=device)
labels = torch.randint(0, 2, (8,), device=device)
optimizer.zero_grad(set_to_none=True)

with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
    logits = model(features)
    loss = criterion(logits, labels)

if scaler.is_enabled():
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
print(loss.detach().item())
```

## 12. 自动选择 AMP 配置

把设备、dtype、开关集中配置，避免到处写死 CUDA。

```python
def build_amp_config(device):
    if device.type == "cuda":
        bf16 = torch.cuda.is_bf16_supported()
        return {
            "enabled": True,
            "dtype": torch.bfloat16 if bf16 else torch.float16,
            "use_scaler": not bf16,
        }
    return {
        "enabled": False,
        "dtype": torch.float32,
        "use_scaler": False,
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_config = build_amp_config(device)
print(amp_config)
```

此配置将 CPU 默认设为 FP32。若 CPU 支持 BF16 且 benchmark 有收益，可以单独启用并验证算子覆盖和稳定性。

## 13. 完整 MLP 训练循环

模型参数保持 FP32；autocast 包住 forward 和 loss；scaler 仅在启用时参与反向更新。

```python
class Classifier(nn.Module):
    def __init__(self, input_dim=4, classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, classes),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
amp_enabled = device.type == "cuda"
amp_dtype = torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
```

一个 epoch 中按样本数累计 loss，避免最后一个小 batch 被错误等权：

```python
def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                    amp_enabled, amp_dtype):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0

    for features, labels in loader:
        features = features.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(features)
            loss = criterion(logits, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        n = labels.size(0)
        loss_sum += loss.detach().item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        count += n

    return loss_sum / count, correct / count
```

验证不调用 scaler，因为没有 backward 和 optimizer.step：

```python
@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp_enabled, amp_dtype):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0

    for features, labels in loader:
        features = features.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(features)
            loss = criterion(logits, labels)
        n = labels.size(0)
        loss_sum += loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        count += n

    return loss_sum / count, correct / count
```

## 14. 梯度裁剪必须先 unscale

scaler 反向后梯度仍被放大。若直接裁剪，阈值含义会错。顺序是 backward、unscale、clip、step、update。

```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
    loss = criterion(model(features), labels)

if scaler.is_enabled():
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
print(grad_norm)
```

一次迭代对同一个 optimizer 只调用一次 unscale_。裁剪不能替代检查输入、学习率和 loss。

## 15. 检查 NaN 和 Inf

GradScaler 能检测非有限梯度并跳过更新，但不会修复输入本身的 NaN 或非法公式。

```python
def check_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")

check_finite("features", features)
check_finite("loss", loss.detach())
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        check_finite(f"grad/{name}", parameter.grad.detach())
```

出现问题时关闭 AMP 用同一 batch 跑 FP32。FP32 也失败则优先查数据、公式、标签和学习率；只有 AMP 失败时再定位低精度范围或敏感算子。

```python
torch.autograd.set_detect_anomaly(True)
```

anomaly detection 很慢，只用于定位问题，不应长期打开。

## 16. 梯度累积

梯度累积把多个 micro-batch 合并为一次参数更新，适合显存受限情况。通常每个 micro-batch 的 loss 除以 accumulation_steps。scaler 对每个 micro-batch 做 scale/backward，但只在累积完成时 step/update。

```python
accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)

for micro_step, (features, labels) in enumerate(train_loader):
    features = features.to(device)
    labels = labels.to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        loss = criterion(model(features), labels) / accumulation_steps

    if scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()

    should_step = (micro_step + 1) % accumulation_steps == 0
    if should_step:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

epoch 末尾不足 accumulation_steps 的 micro-batch 应另行处理；按固定步数缩放还是按实际样本数归一化，必须明确。

## 17. AMP checkpoint

续训需保存模型、optimizer、GradScaler、scheduler、epoch、global step 和 AMP 配置。否则恢复后动量或 loss scale 变化，训练轨迹可能不连续。

```python
checkpoint = {
    "epoch": epoch,
    "global_step": global_step,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
    "amp_enabled": amp_enabled,
    "amp_dtype": str(amp_dtype),
}
torch.save(checkpoint, "checkpoint_amp.pt")
```

恢复时先构造相同结构，再加载状态：

```python
checkpoint = torch.load("checkpoint_amp.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
if scaler.is_enabled() and checkpoint.get("scaler") is not None:
    scaler.load_state_dict(checkpoint["scaler"])
start_epoch = checkpoint["epoch"] + 1
```

推理通常只需模型 state；续训需要完整训练状态。

## 18. 局部禁用 autocast

某些自定义公式或第三方算子可能数值敏感，可局部禁用 autocast，并将输入转为 FP32。禁用范围应尽量小。

```python
with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
    hidden = model.net[0](features)
    with torch.autocast(device_type=device.type, enabled=False):
        stable_value = hidden.float().log1p().sum()
    output = model.net[2](hidden)
print(stable_value, output.shape)
```

不要因为一个算子不稳定就关闭整个训练的 AMP；先定位具体操作并比较结果。

## 19. 哪些运算可能保留 FP32

autocast 通常让适合加速的线性层、卷积等使用低精度，让部分归约、指数、对数、归一化或损失计算保持 FP32。具体规则随设备和版本变化。

```python
x = torch.randn(16, 32, device=device)
layer = nn.Linear(32, 64).to(device)
with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
    hidden = layer(x)
    total = hidden.sum()
print("hidden:", hidden.dtype, "total:", total.dtype)
```

不要根据一次打印结果硬编码算子 dtype。数值行为应通过目标设备测试。

## 20. FP16 与 BF16 选择

可先按以下方式尝试：

| 环境 | 起点 | 备注 |
| --- | --- | --- |
| CUDA 不支持 BF16 | FP16 + GradScaler | 关注溢出和下溢 |
| CUDA 支持 BF16 | BF16 autocast | 通常不需要 scaler，仍需验证 |
| CPU 支持 BF16 | 单独 benchmark | 不保证比 FP32 快 |
| MPS | 查当前版本支持 | 不要照搬 CUDA 假设 |
| 普通 CPU | FP32 | 先确保基线正确 |

```python
def choose_amp(device):
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        return True, torch.float16, True
    return False, torch.float32, False

print(choose_amp(device))
```

最终要比较收敛、验证指标、吞吐、峰值显存和失败率，不能只看 dtype 是否变化。

## 21. 验证阶段的精度

验证不需要 GradScaler，因为没有 backward 和 optimizer.step。可以对验证 forward 使用 autocast 来节省显存，也可以用 FP32 作为稳定基线。应保持指标计算方式一致。

```python
model.eval()
with torch.inference_mode():
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = model(features)
        valid_loss = criterion(logits, labels)
print(valid_loss.item())
```

将结果交给 CPU、NumPy 或指标库前，可显式转为 float32：

```python
predictions = logits.detach().float().cpu()
print(predictions.dtype, predictions.device)
```

## 22. 梯度累积、裁剪和更新时机

梯度累积时，只在真正更新 optimizer 的边界做 unscale、裁剪、step、update 和 zero_grad。

```python
if scaler.is_enabled():
    scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
if scaler.is_enabled():
    scaler.step(optimizer)
    scaler.update()
else:
    optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

学习率调度器应按其设计在 epoch 或 optimizer step 后更新；不要按 micro-batch 更新一个按有效 batch 定义的 scheduler。

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
print(optimizer.param_groups[0]["lr"])
```

## 23. 性能 benchmark

CUDA kernel 异步执行，简单用 Python wall clock 计时可能只测到任务提交时间。benchmark 应 warmup，并使用 CUDA Event 或 synchronize。FP32 与 AMP 必须用相同输入和迭代次数。

```python
if torch.cuda.is_available():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        result = features @ features.T
    end.record()
    torch.cuda.synchronize()
    print("milliseconds:", start.elapsed_time(end))
```

测量前清理显存统计并预热，可观察峰值显存：

```python
if torch.cuda.is_available():
    for _ in range(3):
        _ = model(features)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _ = model(features)
    torch.cuda.synchronize()
    print(torch.cuda.max_memory_allocated() / 1024**2, "MiB")
```

小模型、短序列或数据加载瓶颈下，AMP 可能更慢。应报告样本吞吐、step 时间、显存和验证质量。

## 24. DataLoader 与设备传输

AMP 不会自动将 CPU batch 搬到 GPU。输入通常仍以 FP32 传输，labels 保持 long；autocast 再决定模型运算精度。

```python
for features, labels in train_loader:
    features = features.to(device, dtype=torch.float32, non_blocking=True)
    labels = labels.to(device, dtype=torch.long, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = model(features)
        loss = criterion(logits, labels)
    break
```

CUDA 可 benchmark DataLoader 的 pin_memory 和 non_blocking；这不是 MPS/CPU 的通用优化。

```python
from torch.utils.data import DataLoader, TensorDataset

loader = DataLoader(
    TensorDataset(torch.randn(128, 4), torch.randint(0, 3, (128,))),
    batch_size=32,
    pin_memory=(device.type == "cuda"),
)
print(loader.pin_memory)
```

## 25. 标签、索引和 mask dtype

使用 AMP 不代表所有数据都转半精度。CrossEntropyLoss 类别标签通常是 long，Embedding 索引是整数，mask 常为 bool；只有数值特征和激活由 autocast 选择低精度。

```python
labels = torch.randint(0, 3, (8,), device=device, dtype=torch.long)
token_ids = torch.randint(0, 100, (8, 16), device=device, dtype=torch.long)
mask = token_ids.ne(0)
print(labels.dtype, token_ids.dtype, mask.dtype)
```

## 26. MPS、CPU 与 CUDA 的边界

CPU BF16、CUDA FP16/BF16、MPS AMP 的支持和性能并不相同。代码应基于 device.type 配置，不能在 CPU/MPS 上无条件创建 CUDA scaler。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    amp_enabled = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
else:
    amp_enabled = False
    amp_dtype = torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=False)
print(device, amp_enabled, amp_dtype, scaler.is_enabled())
```

某些版本对禁用 scaler 的 device 参数行为可能不同；如需跨设备封装，应针对项目支持的 PyTorch 版本测试构造方式。

## 27. AMP 与 torch.compile

torch.compile 可能与 autocast 组合使用，但编译后端、dtype 和设备支持因版本而异。先验证未编译训练，再单独加入 compile 并比较首次编译成本与稳态速度。

```python
if hasattr(torch, "compile"):
    compiled_model = torch.compile(model)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        output = compiled_model(features)
    print(output.shape)
```

不要同时改变模型、dtype、batch 和编译选项，否则很难定位回归原因。

## 28. 数值一致性测试

AMP 舍入使 FP32 和 AMP 每步结果通常不完全相同。可以比较相同 batch 的输出差异，但容差要依据任务与 dtype 设置。

```python
model.eval()
with torch.no_grad():
    fp32_output = model(features.float())
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        amp_output = model(features.float())
difference = (fp32_output.float() - amp_output.float()).abs().max()
print("max abs difference:", difference.item())
```

更重要的是在独立验证集比较收敛曲线和最终指标，不能只比较一个 batch。

```python
torch.testing.assert_close(
    fp32_output.float(),
    amp_output.float(),
    rtol=1e-2,
    atol=1e-2,
)
```

容差仅作演示，不能机械复制到所有模型。

## 29. 常见故障排查

```text
现象                         检查方向
loss NaN/Inf                 输入、公式、学习率、FP16 溢出、scaler
AMP 比 FP32 慢                设备支持、batch、warmup、数据传输、同步
显存没有明显下降              是否启用 autocast、激活占比、batch 是否变大
梯度裁剪无效                  clip 前是否 unscale_
dtype/device mismatch         模型、输入、标签、mask 是否兼容
恢复训练不连续                scaler、optimizer、scheduler 是否恢复
验证曲线波动                  eval/inference_mode 和数据管线是否正确
```

用同一个小 batch 依次测试 FP32、autocast、autocast+scaler，能缩小问题范围。

```python
def finite_or_raise(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")

finite_or_raise("features", features)
finite_or_raise("logits", logits)
finite_or_raise("loss", loss.detach())
```

## 30. 完整训练模板

下面模板把 autocast、scaler、裁剪和指标统计整合起来。模型输出 logits [B,C]，labels 是 long [B]。

```python
def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, amp_enabled, amp_dtype, max_grad_norm=None):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0

    for features, labels in loader:
        features = features.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(features)
            loss = criterion(logits, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        n = labels.size(0)
        loss_sum += loss.detach().float().item() * n
        correct += (logits.detach().argmax(1) == labels).sum().item()
        count += n

    return loss_sum / count, correct / count
```

验证阶段关闭梯度，不使用 scaler：

```python
@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp_enabled, amp_dtype):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0

    for features, labels in loader:
        features = features.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(features)
            loss = criterion(logits, labels)
        n = labels.size(0)
        loss_sum += loss.float().item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        count += n

    return loss_sum / count, correct / count
```

根据设备准备配置，再调用训练/验证函数：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp_enabled = True
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
else:
    amp_dtype = torch.float32
    amp_enabled = False
    scaler = torch.amp.GradScaler("cuda", enabled=False)

print(amp_enabled, amp_dtype, scaler.is_enabled())
```

## 31. 超参数与日志

建议记录 AMP dtype、scaler scale、loss、学习率、吞吐和峰值显存。这样可以判断 AMP 是否启用、是否频繁跳步、是否真正提升效率。

```python
if scaler.is_enabled():
    print("loss scale:", scaler.get_scale())
print("learning rate:", optimizer.param_groups[0]["lr"])
```

若使用 TensorBoard，可记录指标和配置；不要把敏感文本或凭证写入 event 文件。

```python
writer.add_scalar("loss/train", train_loss, epoch)
writer.add_scalar("loss/valid", valid_loss, epoch)
writer.add_text("config/amp", str({"enabled": amp_enabled, "dtype": str(amp_dtype)}), 0)
```

## 32. 生产前验证清单

在正式使用 AMP 前：

- FP32 基线先正确收敛；
- AMP 下 loss、梯度和输出均为有限值；
- prediction/target shape、标签 dtype 正确；
- 梯度裁剪在 unscale 之后；
- 验证指标和测试流程独立；
- benchmark 有 warmup，CUDA 计时同步；
- 比较吞吐、峰值显存、验证质量和最终效果；
- 断点保存 scaler、optimizer、scheduler；
- 记录硬件、PyTorch、dtype 和预处理配置。

```python
checks = {
    "amp_enabled": amp_enabled,
    "finite_loss": torch.isfinite(loss.detach()).item(),
    "shape_checked": logits.shape[0] == labels.shape[0],
    "device": str(device),
}
print(checks)
```

## 33. 总结

混合精度的核心不是把所有 Tensor 改成半精度，而是让适合的运算使用低精度，同时保留必要的 FP32 数值稳定性。

```text
FP32 参数与优化器状态
  + autocast 选择部分低精度前向
  + FP16 时 GradScaler 保护反向梯度
  + unscale 后裁剪和检查梯度
  + 验证、保存状态并公平 benchmark
```

工程原则：

- 先建立正确的 FP32 baseline，再启用 AMP；
- 用 torch.autocast 管理计算上下文；
- FP16 训练通常配合 torch.amp.GradScaler；
- 梯度裁剪前先 unscale；
- 标签、索引和 mask 保持各自的 dtype 契约；
- 验证不需要 scaler；
- 续训保存 scaler、optimizer 和 scheduler 状态；
- 以相同数据比较速度、显存、稳定性和验证指标；
- CPU、CUDA、MPS 的支持与收益不能相互假设。

只有模型、数据、损失、设备、日志和评估保持一致，低精度的效率收益才有实际意义。
