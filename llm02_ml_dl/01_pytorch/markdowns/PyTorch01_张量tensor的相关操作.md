# PyTorch 张量 tensor 的相关操作

本文从张量的基本概念开始，依次介绍创建、访问、计算、形状变换、NumPy 互操作、自动求导，以及模型训练中常见的设备和高级操作。示例基于 PyTorch 2.x；除特别说明外，代码默认已执行 `import torch`。

## 1. 张量是什么

张量（Tensor）是 PyTorch 中承载数据的核心对象。可以把它看作带有数据类型、形状和设备信息的多维数组：

| 概念 | 示例 | 含义 |
| --- | --- | --- |
| 标量 | `torch.tensor(3.0)` | 0 维张量 |
| 向量 | `torch.tensor([1, 2, 3])` | 1 维张量 |
| 矩阵 | `torch.zeros(2, 3)` | 2 维张量 |
| 高维张量 | `torch.randn(2, 3, 4)` | 可表达批次、通道、序列、特征等维度 |

图像批次常用 `[N, C, H, W]`，文本批次常见 `[N, L, D]`：分别表示批次、通道/序列长度、空间尺寸/特征维度。维度本身没有固定语义，语义由具体任务约定。

## 2. 创建张量与查看属性

### 2.1 从 Python 数据创建

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x)
print(x.shape)   # torch.Size([2, 3])
print(x.ndim)    # 2
print(x.numel()) # 元素总数：6
print(x.dtype)   # torch.int64
print(x.device)  # cpu
```

`torch.tensor(...)` 会复制输入数据并推断类型。浮点列表通常得到默认浮点类型，整数列表通常得到 `torch.int64`。需要确定精度时，显式传入 `dtype`：

```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
labels = torch.tensor([0, 1, 2], dtype=torch.long)
```

常见类型包括 `torch.float32`、`torch.float64`、`torch.float16`、`torch.bfloat16`、`torch.int64`、`torch.int32` 和 `torch.bool`。分类标签和索引通常使用 `long`；模型输入和参数通常使用浮点类型。

### 2.2 常用工厂函数

```python
zeros = torch.zeros(2, 3)                 # 全 0
ones = torch.ones(2, 3)                   # 全 1
empty = torch.empty(2, 3)                 # 未初始化，读取前必须写入
full = torch.full((2, 3), 7.0)            # 填充值
arange = torch.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, steps=5)  # 等间隔，包含端点
rand = torch.rand(2, 3)                   # [0, 1) 均匀分布
normal = torch.randn(2, 3)                # 标准正态分布
```

`empty` 只分配内存，不会将元素清零，因此不要在填充前使用其中的值。需要与已有张量保持 dtype/device 时，优先使用 `*_like`：

```python
x = torch.randn(2, 3, dtype=torch.float64)
y = torch.zeros_like(x)  # 同形状、同 dtype、同 device
z = torch.ones_like(x, dtype=torch.float32)
```

### 2.3 随机种子

```python
torch.manual_seed(42)
a = torch.rand(3)

torch.manual_seed(42)
b = torch.rand(3)
assert torch.equal(a, b)
```

固定种子有助于复现实验，但不同硬件、PyTorch 版本或非确定性算子仍可能产生差异。GPU 上的确定性设置还需要根据具体算子单独处理。

## 3. 访问、切片与条件选择

PyTorch 的索引语法与 Python/NumPy 类似，索引从 0 开始：

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

x[0, 1]       # 第 0 行第 1 列，标量张量
x[1]          # 第 1 行
x[:, 2]       # 所有行的第 2 列
x[:2, 1:3]    # 前两行、第 1 和第 2 列
x[..., -1]    # 任意前置维度上的最后一列
```

整数索引会移除对应维度；若要保留维度，可用切片或 `select` 后的 `unsqueeze`：

```python
row = x[1:2, :]  # shape: [1, 4]
col = x[:, 2:3]  # shape: [3, 1]
```

布尔掩码适合筛选元素：

```python
x = torch.tensor([1, -2, 3, -4])
mask = x > 0
positive = x[mask]       # tensor([1, 3])
x[x < 0] = 0              # 满足条件的位置原地赋值
```

高级索引与基本切片的内存行为并不完全相同。不要依赖索引结果是否共享底层存储；若要明确复制，请调用 `.clone()`。

## 4. 张量运算

### 4.1 逐元素运算

形状相同的张量可进行逐元素加减乘除：

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])

a + b       # tensor([11., 22., 33.])
a - b
a * b       # 逐元素乘法，不是矩阵乘法
a / b
a ** 2
torch.sqrt(a)
torch.exp(a)
torch.log(a)
torch.abs(torch.tensor([-1.0, 2.0]))
```

比较运算返回布尔张量：`a > 0`、`a == b`、`a != b`。逻辑组合可使用 `&`、`|`、`~`，条件选择可使用 `torch.where`：

```python
x = torch.tensor([-2.0, 0.5, 3.0])
y = torch.where(x > 0, x, torch.zeros_like(x))
```

### 4.2 广播机制

当两个张量形状不同，PyTorch 会从尾部维度开始比较；每一维必须相等，或其中一个为 1，缺失维度视为 1。符合条件时，小维度会被逻辑扩展，而不是实际复制数据：

```python
matrix = torch.zeros(2, 3)  # [2, 3]
row = torch.tensor([1.0, 2.0, 3.0])  # [3]
result = matrix + row  # row 广播到 [2, 3]

column = torch.tensor([[10.0], [20.0]])  # [2, 1]
result = matrix + column  # column 广播到 [2, 3]
```

常见的 `[B, D]` 与 `[D]` 运算会对批次中每个样本使用同一组特征值。若广播报错，先把各操作数的 `shape` 写出来，逐维从右向左核对。

### 4.3 矩阵乘法与线性代数

```python
a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = a @ b                 # [2, 4]
c = torch.matmul(a, b)    # 与 @ 等价

v = torch.tensor([1.0, 2.0, 3.0])
dot = torch.dot(v, v)      # 向量点积，标量
```

`*` 是逐元素乘法，`@`/`matmul` 是矩阵乘法。`matmul` 支持批量矩阵乘法，会对前置批次维度进行广播。常用线性代数函数还包括：

```python
m = torch.randn(3, 3)
m_t = m.T
inverse = torch.linalg.inv(m)          # 方阵可逆时求逆
values, vectors = torch.linalg.eig(m)  # 特征值与特征向量
norm = torch.linalg.norm(m)
```

实际求解线性方程时，通常用 `torch.linalg.solve(A, b)`，比显式求 `inv(A) @ b` 更稳定、更高效。

### 4.4 聚合、统计与维度参数

```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x.sum()              # 所有元素求和
x.mean()             # 全局平均
x.max()
x.min()
x.argmax()           # 展平后的最大值索引

x.sum(dim=0)         # 沿第 0 维聚合，结果 shape [2]
x.mean(dim=1)        # 沿第 1 维聚合，结果 shape [2]
x.sum(dim=1, keepdim=True)  # shape [2, 1]
```

`dim` 指被消去/聚合的轴。`keepdim=True` 会保留长度为 1 的维度，常用于后续广播。返回最大值及其索引可用：

```python
values, indices = x.max(dim=1)
```

## 5. 形状变化

形状变换常用于把数据整理成算子或模型需要的布局。下面 `x` 包含 24 个元素：

```python
x = torch.arange(24)
a = x.reshape(2, 3, 4)
b = x.view(2, 3, 4)
```

### 5.1 `reshape` 与 `view`

`reshape` 在元素总数兼容时改变形状；如果内存布局允许，它可能返回共享存储的视图，否则可能复制。`view` 要求底层内存布局兼容，不满足时会报错。一般优先使用 `reshape`；当明确需要视图语义时再用 `view`。

```python
x = torch.arange(12).reshape(3, 4)
x.reshape(2, 6)
x.reshape(-1)       # -1 由 PyTorch 推断，此处为 12
```

元素总数必须保持一致，例如 12 个元素可变成 `[3, 4]` 或 `[2, 2, 3]`，不可变成 `[5, 3]`。

### 5.2 增加、删除维度

```python
x = torch.arange(6).reshape(2, 3)
x.unsqueeze(0).shape  # [1, 2, 3]
x.unsqueeze(-1).shape # [2, 3, 1]

y = x.unsqueeze(0)   # [1, 2, 3]
y.squeeze(0).shape    # [2, 3]
y.squeeze().shape     # 移除所有长度为 1 的维度
```

尽量给 `squeeze` 指定维度，避免批次大小恰好为 1 时意外移除批次轴。

### 5.3 转置与维度重排

```python
x = torch.randn(2, 3, 4)
x.transpose(1, 2).shape  # [2, 4, 3]，交换两个维度
x.permute(2, 0, 1).shape # [4, 2, 3]，按给定顺序重排全部维度
```

转置或 `permute` 通常返回非连续视图。若后续算子需要连续内存，可调用 `.contiguous()`；但不必无条件调用，因为它可能产生一次复制。

```python
y = x.permute(0, 2, 1).contiguous()
z = y.view(2, 12)
```

### 5.4 拼接、堆叠与拆分

```python
a = torch.ones(2, 3)
b = torch.zeros(2, 3)

torch.cat([a, b], dim=0).shape    # [4, 3]，沿已有维度连接
torch.cat([a, b], dim=1).shape    # [2, 6]
torch.stack([a, b], dim=0).shape  # [2, 2, 3]，新增一个维度

parts = torch.chunk(torch.arange(8), chunks=3)  # 尽量均分，数量可能不足 chunks
parts = torch.split(torch.arange(8), [2, 3, 3]) # 按指定长度拆分
```

`cat` 要求除拼接轴外其他维度相同；`stack` 要求所有输入形状完全相同。

## 6. PyTorch 与 NumPy 互操作

NumPy 适合数据处理和科学计算，PyTorch 张量适合设备加速和自动求导。CPU 上的 `from_numpy`/`numpy()` 通常共享内存：一方修改数据，另一方也会看到变化。

```python
import numpy as np

array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
t = torch.from_numpy(array)  # CPU 张量，与 array 共享内存
t[0] = 99
print(array[0])  # 99.0

back = t.numpy()  # CPU、且不需要梯度的张量可转换为 ndarray
back[1] = 88
print(t[1])       # tensor(88.)
```

显式复制可避免共享：

```python
t_copy = torch.tensor(array)       # 复制
array_copy = t.detach().numpy().copy()
```

GPU 张量不能直接调用 `.numpy()`；需要先移到 CPU。含梯度的张量也不能直接转为 NumPy，因为 NumPy 不跟踪 PyTorch 的计算图：

```python
gpu_tensor = t.to("cuda") if torch.cuda.is_available() else t
array = gpu_tensor.detach().cpu().numpy()
```

将 NumPy 数组转为张量时，若 dtype 不符合模型要求，显式转换：

```python
x = torch.from_numpy(np_array).to(dtype=torch.float32)
```

## 7. 自动求导 autograd

自动求导会记录参与运算的张量操作，并按链式法则计算梯度。模型训练通常通过损失函数对参数求梯度，再由优化器更新参数。

### 7.1 标量函数求导

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()

print(y)       # tensor(10., grad_fn=...)
print(x.grad)  # tensor(7.)，dy/dx = 2x + 3，在 x=2 处为 7
```

默认情况下，只有叶子张量的 `.grad` 会被保存。由其他运算得到的中间结果通常不是叶子张量；需要检查其中间梯度时可调用 `retain_grad()`。

### 7.2 向量输出与反向传播

`backward()` 默认要求输出为标量。若输出是向量，需要传入向量-雅可比积中的向量，或先将输出规约为标量：

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.sum().backward()
print(x.grad)  # tensor([2., 4., 6.])
```

### 7.3 梯度累积与清零

多次调用 `backward()` 时，梯度会累加到 `.grad`，这对分批累积梯度有用，但普通训练循环必须在每次参数更新前清零：

```python
x = torch.tensor(2.0, requires_grad=True)
(x ** 2).backward()
(3 * x).backward()
print(x.grad)  # 7.0，而不是只得到 4.0
x.grad = None  # 也可使用 optimizer.zero_grad(set_to_none=True)
```

清零优化器梯度的常见写法是 `optimizer.zero_grad(set_to_none=True)`，随后计算新损失并执行反向传播。

### 7.4 停止跟踪与分离计算图

推理或只做数据计算时，通常不需要构建计算图：

```python
with torch.no_grad():
    prediction = model(inputs)

with torch.inference_mode():
    prediction = model(inputs)
```

`inference_mode` 通常有更少的额外开销，但其中创建的张量不适合之后再参与需要 autograd 的计算。若只想让某个结果脱离原计算图，可使用 `detach()`：

```python
y = some_tensor.detach()
```

`detach()` 与原张量共享存储；若需要独立数据副本，使用 `some_tensor.detach().clone()`。

### 7.5 原地操作注意事项

以 `_` 结尾的许多操作会原地修改张量，例如 `add_()`、`zero_()`。对参与计算图的叶子张量做原地修改，可能破坏反向传播所需的历史值并触发错误。更新参数时应放在 `torch.no_grad()` 中，或交由优化器完成：

```python
w = torch.tensor(2.0, requires_grad=True)
loss = w ** 2
loss.backward()
with torch.no_grad():
    w -= 0.1 * w.grad
```

除非确有内存或性能需求，初学时优先使用非原地运算。

## 8. dtype、设备与模型训练的基础流程

### 8.1 CPU 与 GPU/MPS

张量和参与同一运算的张量通常必须位于同一设备。常用设备选择方式：

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

x = torch.randn(2, 3).to(device)
y = torch.zeros_like(x)  # 自动继承 x 的 device
```

模型也需要迁移到相同设备：`model = model.to(device)`。可通过 `tensor.device` 和 `next(model.parameters()).device` 检查设备。

### 8.2 转换类型与设备

```python
x = torch.tensor([1, 2, 3])
x_float = x.to(dtype=torch.float32)
x_long = x.long()
x_cpu = x.cpu()
x_device = x.to(device)
```

`.to(...)` 会返回转换后的张量；除非使用原地版本，不要假设旧变量已被修改。传入 `device` 或 `dtype` 可一次完成相应转换。

### 8.3 一个最小训练步骤

```python
import torch.nn as nn

model = nn.Linear(3, 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

inputs = torch.randn(8, 3, device=device)
targets = torch.randn(8, 1, device=device)

model.train()                         # 设置训练模式
optimizer.zero_grad(set_to_none=True) # 清理上一步梯度
predictions = model(inputs)           # 前向计算
loss = criterion(predictions, targets)
loss.backward()                       # 计算参数梯度
optimizer.step()                      # 更新参数
```

训练循环的核心顺序是：清梯度、前向计算、计算损失、反向传播、更新参数。使用 `nn.Module` 构建的模型在训练/推理时还应分别调用 `model.train()` 和 `model.eval()`；它们会影响 Dropout、BatchNorm 等层的行为。

## 9. 常见高级张量操作

### 9.1 排序、Top-K 与唯一值

```python
x = torch.tensor([3.0, 1.0, 4.0, 2.0])
sorted_values, sorted_indices = torch.sort(x)
values, indices = torch.topk(x, k=2)
unique = torch.unique(torch.tensor([1, 1, 2, 3]))
```

在分类任务中，常用 `logits.topk(k, dim=-1)` 取得每个样本得分最高的类别。

### 9.2 `gather`：按索引从指定维度取值

`gather` 的输入和索引通常具有相同维度数；除指定维度外，索引在其他维度不能超过输入范围：

```python
scores = torch.tensor([[0.1, 0.7, 0.2],
                       [0.8, 0.1, 0.1]])  # [batch, classes]
labels = torch.tensor([[1], [0]])         # 每行要提取的类别索引
selected = scores.gather(dim=1, index=labels)
# tensor([[0.7000], [0.8000]])
```

### 9.3 `scatter`：按索引写入

```python
out = torch.zeros(2, 3)
indices = torch.tensor([[1], [2]])
values = torch.tensor([[5.0], [9.0]])
out.scatter_(dim=1, index=indices, src=values)
# tensor([[0., 5., 0.], [0., 0., 9.]])
```

这里使用了原地版本 `scatter_`。索引越界、索引形状与源数据形状不匹配时会报错；多个位置冲突时，不应依赖某些设备上的写入顺序。

### 9.4 掩码填充与裁剪

```python
x = torch.tensor([1.0, float("nan"), 3.0])
clean = torch.nan_to_num(x, nan=0.0)

scores = torch.tensor([-2.0, 0.5, 4.0])
clipped = torch.clamp(scores, min=0.0, max=1.0)
```

序列任务常以布尔掩码标识有效位置，再使用广播、`masked_fill` 或损失函数的 `ignore_index` 忽略填充位置：

```python
logits = torch.randn(2, 4, 5)  # [batch, length, classes]
padding_mask = torch.tensor([[False, False, True, True],
                             [False, False, False, True]])
masked_logits = logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)
```

### 9.5 `einsum` 与维度表达

`torch.einsum` 使用下标表达求和和乘积，适合复杂的批量张量运算。例：

```python
a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = torch.einsum("ik,kj->ij", a, b)  # 矩阵乘法，shape [2, 4]

batch_a = torch.randn(5, 2, 3)
batch_b = torch.randn(5, 3, 4)
batch_c = torch.einsum("bik,bkj->bij", batch_a, batch_b)
```

初学时建议先熟练 `sum`、`matmul`、`transpose` 等基础操作；只有在表达更清晰或能减少中间张量时再使用 `einsum`。

## 10. 保存、加载与检查张量

保存张量或训练状态可使用 `torch.save`：

```python
torch.save({"weights": model.state_dict(), "step": 100}, "checkpoint.pt")
```

加载时指定映射设备；只加载可信来源的文件。较新的 PyTorch 版本支持 `weights_only=True` 加载权重类数据：

```python
checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["weights"])
```

排查张量问题时，优先检查：

```python
print(x.shape, x.dtype, x.device, x.requires_grad)
print(torch.isfinite(x).all())  # 是否全部为有限值
```

## 11. 常见问题速查

| 现象 | 常见原因 | 排查方向 |
| --- | --- | --- |
| `mat1 and mat2 shapes cannot be multiplied` | 矩阵乘法内维不匹配 | 检查 `@` 左侧最后一维与右侧倒数第二维 |
| 张量不能相加/广播失败 | 从右向左比较时有维度既不相等也不为 1 | 打印并逐维核对 `shape`，必要时 `unsqueeze` |
| `view` 报 stride/contiguous 错误 | 张量经过转置等操作后不连续 | 改用 `reshape`，或确认需要后调用 `contiguous()` |
| 期望有梯度但 `.grad` 是 `None` | 未设置 `requires_grad`、不是叶子张量、或计算在 no-grad 中 | 检查 `requires_grad`、`is_leaf`、执行路径 |
| 第二次反向传播失败 | 计算图默认在反向传播后释放 | 重新执行前向计算；仅在需要时使用 `retain_graph=True` |
| GPU 运算提示设备不一致 | 输入、参数或索引位于不同设备 | 将模型和张量统一 `.to(device)`；索引也检查 device 要求 |
| 梯度越来越大 | 每步没有清零，梯度被累加 | 每个更新步骤调用 `optimizer.zero_grad(...)` |

## 12. 建议的练习顺序

1. 创建标量、向量、矩阵，观察 `shape`、`ndim`、`dtype` 和 `device`。
2. 用切片和布尔掩码提取数据，并验证每一步结果的形状。
3. 分别用 `*` 和 `@` 完成逐元素乘法与矩阵乘法，解释结果形状。
4. 将 `[B, L, D]` 的张量转为 `[B, D, L]`，再沿序列维求平均。
5. 将 NumPy 数组转为张量，实验共享内存与显式复制的差别。
6. 对 `y = (x - 3) ** 2` 调用 `backward()`，手算并核对梯度。
7. 用 `nn.Linear` 完成一次训练步骤，并检查参数更新前后的差异。

掌握这些操作后，再进入 `Dataset`/`DataLoader`、神经网络层、损失函数、优化器和完整训练循环，会更容易定位形状、类型、设备与梯度方面的问题。

## 13. 神经网络模块 `nn.Module`

PyTorch 通常通过继承 `torch.nn.Module` 定义模型。模型层作为属性注册后，其参数会自动参与梯度计算、设备迁移和 `state_dict` 保存。

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

model = MLP(input_size=10, hidden_size=32, num_classes=3)
logits = model(torch.randn(4, 10))
print(logits.shape)  # [4, 3]
```

`__init__` 中定义层和子模块，`forward` 中描述前向计算。训练时不要直接调用 `model.forward(x)`，而要写 `model(x)`，这样 PyTorch 才能正确执行模块钩子等机制。`nn.Sequential` 适合简单的顺序网络；需要分支、循环或多输入时，可在 `forward` 中显式编写逻辑。

### 13.1 参数与缓冲区

`nn.Parameter` 会作为可训练参数注册；普通张量属性不会自动成为参数。不可训练但需要随模型保存、迁移的状态可以通过 `register_buffer` 注册。

```python
class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer("running_count", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        return x * self.weight

model = Scale()
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

优化器一般接收 `model.parameters()`。冻结某些层时可设置 `parameter.requires_grad_(False)`，并确认优化器中是否只包含需要更新的参数。

## 14. 数据集与批次加载

训练数据通常通过 `Dataset` 描述单个样本如何读取，再交给 `DataLoader` 批量取样、打乱和并行加载。小型内存数据可用 `TensorDataset`：

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

features = torch.randn(100, 10)
labels = torch.randint(0, 3, (100,))
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_features, batch_labels in loader:
    print(batch_features.shape, batch_labels.shape)
    break
```

分类标签的形状通常是 `[batch]`，类型通常为 `torch.long`。回归标签常是浮点张量，形状应与模型输出相匹配，例如预测和目标均为 `[batch, 1]`。

### 14.1 自定义 `Dataset`

数据来自文件或需要按需预处理时，可以实现 `__len__` 和 `__getitem__`：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label
```

`DataLoader` 会调用这些方法组成批次。常用参数：

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,       # 训练集通常打乱
    num_workers=0,      # 先用 0 确认数据读取正确，再按环境调大
    drop_last=False,    # 是否丢弃最后一个不满批次的数据
    pin_memory=torch.cuda.is_available(),
)
```

验证集和测试集通常设置 `shuffle=False`。在多进程加载环境中，入口脚本常需要放在 `if __name__ == "__main__":` 保护下；Windows/macOS 上尤其要留意这一点。`pin_memory=True` 主要对 CUDA 主机到设备的数据传输有帮助，配合 `.to(device, non_blocking=True)` 使用。

## 15. 常见损失函数与输出约定

损失函数把模型输出和目标转换为可优化的标量。选择损失函数时，必须同时核对模型输出的形状、标签形状和数据类型。

| 任务 | 模型输出 | 目标 | 常用损失 |
| --- | --- | --- | --- |
| 回归 | 浮点预测值 | 同形状浮点值 | `nn.MSELoss`、`nn.L1Loss`、`nn.SmoothL1Loss` |
| 单标签多分类 | 未经过 softmax 的 logits `[N, C]` | 类别索引 `[N]`，`long` | `nn.CrossEntropyLoss` |
| 二分类/多标签 | 未经过 sigmoid 的 logits | 浮点 0/1，通常与 logits 同形状 | `nn.BCEWithLogitsLoss` |

### 15.1 回归损失

```python
criterion = nn.MSELoss()
prediction = torch.tensor([[2.0], [4.0]])
target = torch.tensor([[1.0], [5.0]])
loss = criterion(prediction, target)
```

MSE 会平方误差，较大的偏差影响更明显；`L1Loss` 使用绝对误差；`SmoothL1Loss` 在小误差附近更平滑，常用于检测等任务。

### 15.2 单标签多分类

`CrossEntropyLoss` 内部会处理 softmax/log-softmax，因此传入原始 logits，不要先做 `softmax`：

```python
logits = torch.randn(8, 4)             # 8 个样本、4 个类别
class_ids = torch.randint(0, 4, (8,))  # long 类型类别索引
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, class_ids)
```

推理时才可用 `logits.argmax(dim=1)` 取预测类别。若目标是 one-hot 格式，应先确认损失函数支持的目标格式，不能把 one-hot 标签误当作类别索引。

### 15.3 二分类与多标签分类

```python
logits = torch.randn(8, 1)
targets = torch.randint(0, 2, (8, 1)).float()
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets)
```

`BCEWithLogitsLoss` 将 sigmoid 与二元交叉熵合并，数值上比手动先 sigmoid 再计算 BCE 更稳定。多标签任务可把输出和目标都设为 `[N, 类别数]`，每个类别独立表示是否成立。

## 16. 优化器与学习率

优化器根据 `.grad` 更新模型参数。常用优化器包括 SGD 和 Adam/AdamW：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 或
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

学习率决定每次参数更新的幅度，是最重要的训练超参数之一。过大可能使损失震荡或发散，过小则可能收敛很慢。`weight_decay` 是常见正则化选项；AdamW 将权重衰减与梯度更新解耦。

学习率调度器按训练进程调整学习率：

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(num_epochs):
    train_one_epoch(...)
    scheduler.step()  # StepLR 通常每个 epoch 结束后调用
```

不同 scheduler 的 `step()` 时机并不完全相同：按 epoch 调度的通常在 epoch 后调用，按验证指标调度的如 `ReduceLROnPlateau` 则应传入指标。可通过 `optimizer.param_groups[0]["lr"]` 查看当前学习率。

## 17. 完整的训练、验证与推理流程

下面用合成的三分类数据串起数据加载、模型、损失、优化器、训练循环、验证和预测。数据仅用于演示，不代表真实任务效果。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构造可分的二维合成数据。
centers = torch.tensor([[-2.0, -2.0], [2.0, -2.0], [0.0, 2.0]])
labels = torch.arange(3).repeat_interleave(200)
features = centers[labels] + 0.6 * torch.randn(len(labels), 2)

train_x, valid_x = features[:480], features[480:]
train_y, valid_y = labels[:480], labels[480:]
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=64)

model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 3),  # 输出原始 logits
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for epoch in range(20):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_samples += batch_size

    train_loss = total_loss / total_samples
    train_accuracy = total_correct / total_samples

    model.eval()
    valid_correct = 0
    valid_samples = 0
    with torch.inference_mode():
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            valid_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            valid_samples += batch_y.size(0)

    valid_accuracy = valid_correct / valid_samples
    print(
        f"epoch={epoch + 1:02d} "
        f"train_loss={train_loss:.4f} "
        f"train_acc={train_accuracy:.3f} "
        f"valid_acc={valid_accuracy:.3f}"
    )

# 推理：eval 模式 + inference_mode；输出 logits，再取类别索引。
model.eval()
new_samples = torch.tensor([[0.0, 2.0], [-2.0, -2.0]], device=device)
with torch.inference_mode():
    predicted_classes = model(new_samples).argmax(dim=1)
print(predicted_classes.cpu())
```

几个值得记住的细节：

- 训练准确率要用更新前的 `logits` 统计，样本数不固定时按实际 batch 大小累计。
- 平均损失应按样本数加权，不能简单平均各批次损失，否则最后一个较小批次权重会过大。
- 验证/推理时同时使用 `model.eval()` 和 `torch.inference_mode()`；前者切换 Dropout/BatchNorm 行为，后者关闭 autograd 记录。
- 验证数据不参与 `backward()` 和 `optimizer.step()`。

## 18. 验证指标、梯度裁剪与早停

验证集用于观察模型对未参与参数更新的数据的表现。分类准确率可按预测正确数除以样本数计算；不均衡分类还应查看精确率、召回率、F1 等指标。不要只根据训练损失判断泛化能力。

循环网络或较深模型有时会出现梯度过大，可在 `backward()` 后、`optimizer.step()` 前裁剪梯度：

```python
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

梯度裁剪主要限制更新前的梯度范数，不会修复错误的模型、学习率或数据问题。早停（early stopping）则是在验证指标长期没有改善时结束训练；保存当前最佳权重，而不是默认把最后一轮当成最佳模型：

```python
best_valid_loss = float("inf")

if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), "best_model.pt")
```

实际训练中可以设置耐心轮数（patience），允许验证指标短期波动；早停判断应使用与任务一致且稳定的验证指标。

## 19. 模型保存、加载与断点续训

### 19.1 保存和加载模型权重

推荐保存 `state_dict`，并在加载前创建相同结构的模型：

```python
torch.save(model.state_dict(), "model_weights.pt")

new_model = MLP(input_size=10, hidden_size=32, num_classes=3).to(device)
state = torch.load(
    "model_weights.pt",
    map_location=device,
    weights_only=True,
)
new_model.load_state_dict(state)
new_model.eval()
```

PyTorch 的 checkpoint 文件可以包含可执行反序列化内容，因此只加载可信来源的文件。旧版本若不支持 `weights_only` 参数，应查阅所用版本文档并谨慎加载。

### 19.2 保存优化器与训练进度

只保存模型权重适合推理和迁移；希望恢复训练时，还应保存优化器状态、轮次等信息：

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_valid_loss": best_valid_loss,
}
torch.save(checkpoint, "training_checkpoint.pt")
```

恢复时先创建同结构模型和优化器，再加载各自状态：

```python
checkpoint = torch.load(
    "training_checkpoint.pt",
    map_location=device,
    weights_only=True,
)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
best_valid_loss = checkpoint["best_valid_loss"]
```

若使用学习率调度器、混合精度 `GradScaler` 或随机采样器，也要保存对应状态，才能更完整地延续训练。断点续训代码还需处理 checkpoint 不存在、文件损坏等情况。

## 20. 混合精度训练与数据传输

混合精度可以在受支持的设备上使用较低精度执行部分运算，以降低显存占用并提高吞吐量。先在 CUDA 上使用自动混合精度；CPU/MPS 的支持和收益依 PyTorch 版本与硬件而异。

```python
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device, non_blocking=(device.type == "cuda"))
    batch_y = batch_y.to(device, non_blocking=(device.type == "cuda"))

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=(device.type == "cuda"),
    ):
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

初学时先确保普通精度训练正确，再比较混合精度的速度、显存和数值稳定性。不是所有算子或任务都会变快；遇到 NaN、溢出或精度下降时，可先关闭混合精度定位问题。

## 21. 训练代码常见排查清单

1. 打印一个批次的 `features.shape`、`labels.shape`、dtype 和 device。
2. 对照损失函数要求核对模型最后一层输出和标签格式，特别是分类标签是否为 `long`。
3. 确认每轮都调用 `model.train()`；验证前调用 `model.eval()`，并禁用梯度记录。
4. 确认每次参数更新前清零梯度，且 `loss.backward()` 在 `optimizer.step()` 前。
5. 检查损失是否为有限数：`torch.isfinite(loss)`；若异常，再检查输入、标签、学习率和梯度。
6. 记录训练集与验证集指标；训练集持续改善而验证集变差，可能是过拟合。
7. 先用很小的一批数据尝试过拟合。如果连小批数据都无法拟合，优先检查标签、损失、输出维度、梯度流和参数是否进入优化器。

完成本节后，可进一步学习卷积网络、序列模型、注意力机制、迁移学习和自定义训练组件；这些主题都建立在本教程介绍的张量形状、模块、数据加载和训练循环基础上。
