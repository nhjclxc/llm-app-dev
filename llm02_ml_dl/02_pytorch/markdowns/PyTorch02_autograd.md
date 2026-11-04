# PyTorch 自动求导 autograd：从 0 到 1

本文从微积分中的导数出发，逐步说明 PyTorch 如何记录张量运算、构建计算图、计算并管理梯度，最后将这些概念放进神经网络训练流程。示例基于 PyTorch 2.x，默认已执行：

```python
import torch
from torch import nn
```

学习目标：读完后能够解释 `requires_grad`、`grad_fn`、叶子张量和 `.grad` 的关系；正确调用反向传播；避免梯度累积、计算图释放和原地操作等常见问题；理解训练循环中的梯度清零、参数更新和推理模式。

## 1. 为什么需要自动求导

机器学习模型通常由参数化函数组成。训练的目标是调整参数，使损失函数变小。以一元函数为例：

$
y = f(x) = x^2,\quad \frac{dy}{dx}=2x
$

当 `x = 3` 时，输出是 9，导数是 6。导数描述输入发生微小变化时输出变化的速度。对模型参数求损失的导数，可以知道参数往哪个方向调整会使损失下降。

例如线性模型 `y = wx + b` 的均方误差损失为：

$
L(w,b)=\frac{1}{N}\sum_{i=1}^{N}(wx_i+b-y_i)^2
$

手动推导复杂神经网络中每个参数的导数既繁琐又容易出错。PyTorch 的 autograd 会追踪参与计算的张量运算，并用链式法则自动计算导数。

> autograd 负责计算梯度，不负责决定怎样更新参数。优化器（如 `torch.optim.SGD`）读取参数的梯度并执行更新。

## 2. 从最小例子认识计算图

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

print(y)          # tensor(9., grad_fn=<PowBackward0>)
print(y.grad_fn)  # 类似 <PowBackward0 ...>

y.backward()
print(x.grad)     # tensor(6.)
```

这里发生了四件事：

1. `x` 被标记为需要梯度。
2. 执行 `x ** 2` 时，autograd 记录了从 `x` 到 `y` 的运算。
3. `y.backward()` 从结果向前遍历记录的运算，按链式法则计算导数。
4. 由于 `x` 是需要梯度的叶子张量，结果 `dy/dx = 6` 被写入 `x.grad`。

可将它理解为一张有向计算图：`x → 平方运算 → y`。PyTorch 通常在前向计算时动态构建这张图；每次前向调用可以产生不同的图，因此 Python 的分支和循环可以自然地参与计算。

## 3. 哪些张量会被跟踪

### 3.1 `requires_grad`

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
print(x.requires_grad)  # True

constant = torch.tensor([10.0, 20.0])
print(constant.requires_grad)  # False
```

浮点或复数张量可参与求导；整数张量不能设置 `requires_grad=True`。通常模型的浮点参数需要梯度，输入数据和标签通常不需要。

`requires_grad` 是是否追踪该张量运算的开关。只要运算中至少有一个输入需要梯度，且当前处于 grad mode，输出通常也会被追踪：

```python
x = torch.tensor(2.0, requires_grad=True)
c = torch.tensor(5.0)
y = x * c

print(y.requires_grad)  # True
```

### 3.2 叶子张量、非叶子张量和 `grad_fn`

```python
x = torch.tensor(2.0, requires_grad=True)  # 用户直接创建的叶子张量
y = x * 3                                  # 运算产生的非叶子张量
z = y + 1

print(x.is_leaf, y.is_leaf, z.is_leaf)  # True False False
print(x.grad_fn)  # None
print(y.grad_fn)  # 类似 <MulBackward0 ...>
```

- **叶子张量（leaf tensor）**：通常是用户直接创建、且 `requires_grad=True` 的张量，或 `nn.Parameter`。它没有由本图中某个运算产生的 `grad_fn`。
- **非叶子张量**：由运算产生，带有 `grad_fn`，autograd 用它连接回前面的运算。
- **`grad_fn`**：反向传播所需的运算节点信息，不是梯度本身。
- **`.grad`**：某个张量已计算出的梯度值。默认情况下，`.backward()` 会将梯度保存到需要梯度的叶子张量的 `.grad`；中间非叶子张量的 `.grad` 默认不保留。

如确实需要检查中间结果的梯度，可在反向传播前调用 `retain_grad()`：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
y.retain_grad()
z = y ** 2
z.backward()

print(x.grad)  # dz/dx = 2 * y * 3 = 36
print(y.grad)  # dz/dy = 2 * y = 12
```

日常训练通常只读取模型参数的梯度，不需要对中间激活调用 `retain_grad()`。

### 3.3 `nn.Parameter` 与模型参数

`nn.Parameter` 是会被 `nn.Module` 自动注册的张量参数，默认 `requires_grad=True`：

```python
layer = nn.Linear(3, 1)
for name, parameter in layer.named_parameters():
    print(name, parameter.shape, parameter.requires_grad, parameter.is_leaf)
```

优化器通常接收 `model.parameters()`。被冻结的参数仍属于模型，但不再计算梯度：

```python
for parameter in model.parameters():
    parameter.requires_grad_(False)
```

## 4. 反向传播与梯度的含义

### 4.1 标量损失调用 `.backward()`

当输出是标量时，`.backward()` 可以直接计算它对图中叶子张量的导数：

```python
x = torch.tensor(2.0, requires_grad=True)
loss = (x - 5) ** 2
loss.backward()
print(loss.item())  # 9.0
print(x.grad.item())  # -6.0，即 dloss/dx = 2 * (2 - 5)
```

优化时，若参数为 `w`，常见的梯度下降更新是 `w ← w - learning_rate * dL/dw`。梯度为负表示在当前位置增大 `w` 会降低局部损失；梯度为正则相反。

### 4.2 非标量输出：向量-雅可比积

如果输出不是标量，PyTorch 不能凭空决定应该对多个输出如何求导。此时必须给 `.backward(gradient)` 提供与输出同形状的向量，它表示要计算的加权和的系数：

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2

y.backward(torch.ones_like(y))
print(x.grad)  # tensor([2., 4.])
```

这等价于对 `y.sum()` 反向传播，因为给每个输出分量的权重都是 1。一般而言，给定输出向量 `y` 和 `v`，计算的是向量-雅可比积 `vᵀJ`，不是自动生成完整雅可比矩阵。

另一种常见做法是先将多个损失汇总为标量：

```python
per_item_loss = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = per_item_loss.mean()
loss.backward()
print(per_item_loss.grad)  # 每项的梯度为 1/3
```

### 4.3 批次损失的 `sum` 和 `mean`

若逐样本损失为 `losses`：

- `losses.sum()` 计算总和，梯度规模通常随批次大小增长。
- `losses.mean()` 计算平均值，梯度按元素个数归一化。
- `nn` 损失函数的 `reduction` 常可设为 `"none"`、`"sum"` 或 `"mean"`。

选哪一种应与期望的目标函数及学习率配合。梯度突然随 batch size 改变时，检查损失的 reduction。

### 4.4 手动验证链式法则

```python
x = torch.tensor(2.0, requires_grad=True)
u = x ** 2       # du/dx = 2x = 4
loss = 3 * u + 1  # dloss/du = 3
loss.backward()
print(x.grad)    # 3 * 4 = 12
```

计算图把各局部导数相乘：`dloss/dx = (dloss/du) * (du/dx)`。深层神经网络的反向传播正是将这个过程扩展到大量张量运算。

## 5. 梯度累积：最重要的训练习惯

调用 `.backward()` 时，梯度默认**累加**到 `.grad`，而不是覆盖旧值：

```python
x = torch.tensor(2.0, requires_grad=True)
(x ** 2).backward()
print(x.grad)  # 4
(3 * x).backward()
print(x.grad)  # 7：4 + 3
```

这对于分批累积梯度有用，但一般每次参数更新前都需要清零：

```python
x.grad = None
(x ** 2).backward()
```

优化器负责清理它所管理的参数梯度：

```python
optimizer.zero_grad(set_to_none=True)
```

`set_to_none=True` 会把 `.grad` 设为 `None`，常可减少不必要的内存写入。它与填充零在少数优化器行为细节上有所不同；一般训练流程推荐使用该选项。

如果忘记清梯度，下一批次反向传播会叠加到上一批次的梯度，更新结果便不再对应当前批次的损失。这是初学者最常遇到的问题之一。

## 6. 标准训练循环：把概念串起来

```python
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

x = torch.tensor([[1.0], [2.0], [3.0]])
target = torch.tensor([[3.0], [5.0], [7.0]])

for step in range(100):
    optimizer.zero_grad(set_to_none=True)  # 1. 清理上一轮梯度
    prediction = model(x)                   # 2. 前向计算，构建计算图
    loss = criterion(prediction, target)    # 3. 得到标量损失
    loss.backward()                         # 4. 反向传播，填充参数 .grad
    optimizer.step()                        # 5. 使用梯度更新参数

    if step % 20 == 0:
        print(step, loss.item())
```

记忆顺序：**清梯度 → 前向 → 算损失 → 反向 → 更新**。`optimizer.step()` 不会替你调用 `backward()`；`backward()` 也不会修改参数值。

在更新参数的代码后读取参数 `.grad`，读到的是刚才那次反向传播留下的梯度；下一轮的 `zero_grad` 才会清除它。

## 7. 暂停梯度记录：训练、验证和推理

有时不需要构建计算图。例如验证或部署推理时，只计算输出而不更新参数，可使用梯度模式上下文。

### 7.1 `torch.no_grad()`

```python
model.eval()
with torch.no_grad():
    prediction = model(x)
```

块内的运算不会被 autograd 记录，结果通常 `requires_grad=False`。它可降低计算图相关开销，但如果之后希望从这些运算继续反向传播，就不能在该块中计算它们。

### 7.2 `torch.inference_mode()`

```python
model.eval()
with torch.inference_mode():
    prediction = model(x)
```

这是更偏向纯推理的模式，通常比 `no_grad()` 还能减少额外开销，但限制更严格：在其中创建的张量一般不适合稍后再参与需要 autograd 记录的计算。纯推理优先考虑 `inference_mode()`；若输出后还要继续做需要梯度的工作，使用 `no_grad()` 或正常 grad mode 并按需求选择。

### 7.3 这三个概念不要混淆

| 调用 | 作用 | 是否改变梯度记录 |
| --- | --- | --- |
| `model.train()` | 将模块切换到训练行为，如开启 Dropout、更新 BatchNorm 统计量 | 否 |
| `model.eval()` | 将模块切换到评估行为，如关闭 Dropout、使用 BatchNorm 运行统计量 | 否 |
| `torch.no_grad()` / `inference_mode()` | 控制张量运算是否记录 autograd 信息 | 是 |

验证时常见组合是 `model.eval()` 加 `torch.inference_mode()`。`eval()` 本身不会关闭求导；如忘记上下文，仍可能构建计算图。

## 8. 冻结参数、分离计算图与张量复制

### 8.1 `detach()`：从当前图中分离

```python`
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y_detached = y.detach()

print(y_detached.requires_grad)  # False`
```

`detach()` 返回与当前图断开的张量，常用于将张量转成不参与本次求导的值。它通常与原张量共享底层存储，因此不要通过其中一个结果原地修改数据并期待另一个保持独立。确实需要独立副本时使用 `detach().clone()`。

`tensor.detach().cpu().numpy()` 常用于导出结果到 NumPy；CUDA 张量需要先移到 CPU：

```python
array = prediction.detach().cpu().numpy()
```

### 8.2 冻结一部分网络

```python
for parameter in backbone.parameters():
    parameter.requires_grad_(False)

features = backbone(images)
logits = classifier(features)
loss = criterion(logits, labels)
loss.backward()
```

冻结主干参数后，autograd 不会为它们计算和保存参数梯度，但仍可能需要追踪主干输入到后续可训练层之间的运算，具体取决于输入和运算是否需要梯度。不要仅凭“冻结参数”就把整个前向过程包进 `no_grad()`；若要训练接在主干之后的层，需保证该层前向仍在 grad mode 中正常连接。

### 8.3 创建新叶子张量

若已有张量需要作为独立优化变量，可显式分离并开启梯度：

```python
initial = torch.randn(3)
variable = initial.detach().clone().requires_grad_(True)
print(variable.is_leaf)  # True
```

这也避免对原张量的计算图产生不必要的连接。

## 9. 更灵活的梯度 API

### 9.1 `torch.autograd.grad`

`.backward()` 通常将梯度写到叶子张量的 `.grad`。`torch.autograd.grad` 则直接返回指定输入的梯度，不会把结果累加到这些输入的 `.grad`：

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
(dx,) = torch.autograd.grad(y, x)
print(dx)     # 6
print(x.grad) # None
```

多个输入可以一起请求：`torch.autograd.grad(output, (x, y))`。默认要求 `output` 为标量；非标量仍可通过 `grad_outputs` 指定向量-Jacobian 积的向量。

### 9.2 高阶导数 `create_graph=True`

普通反向传播计算出的梯度通常不再记录为可继续求导的图。要对梯度再次求导，需建立梯度计算图：

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 3
(dy_dx,) = torch.autograd.grad(y, x, create_graph=True)
(d2y_dx2,) = torch.autograd.grad(dy_dx, x)

print(dy_dx)       # 27
print(d2y_dx2)     # 18
```

`create_graph=True` 会增加内存与计算开销，应只在需要高阶导数、梯度惩罚或相关算法时使用。

### 9.3 `retain_graph=True`

第一次反向传播后，autograd 通常释放该计算图保存的中间信息以节省内存。对同一张图再次调用 `.backward()` 可能报“trying to backward through the graph a second time”错误。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
y.backward()  # 第二次使用仍保留的图
```

`retain_graph=True` 是保留旧图，不是清梯度；第二次 `.backward()` 仍会再次累加梯度。多数训练循环不需要它，因为每轮前向都会创建新图。若目标是获得不同输入的梯度，`autograd.grad` 常更直接。避免为了压制报错而无条件保留图，这可能导致显存持续增长。

### 9.4 梯度钩子

需要观察或变换某个张量梯度时，可注册钩子：

```python
x = torch.tensor(2.0, requires_grad=True)
handle = x.register_hook(lambda grad: grad.clamp(-1, 1))
(x ** 2).backward()
print(x.grad)  # 原始梯度 4 被钩子限制为 1
handle.remove()
```

钩子应返回形状兼容的梯度（若要替换梯度），并在不再需要时移除句柄。梯度裁剪更常用优化器工具函数 `torch.nn.utils.clip_grad_norm_` 或 `clip_grad_value_`。

## 10. 原地操作和版本错误

带下划线的操作通常是原地修改，例如 `add_()`、`zero_()`；赋值切片也可能原地修改。autograd 为反向传播保存某些前向值，如果这些值在反向前被覆盖，可能出现版本计数错误，也可能让图的梯度语义变得不清晰。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
# 不要尝试对需要梯度的叶子 x 原地写入：x.add_(1)
```

需要修改参数值但不记录为可微运算时，应将更新放进 `torch.no_grad()`：

```python
parameter = nn.Parameter(torch.tensor(1.0))
with torch.no_grad():
    parameter.add_(0.1)
```

实际训练通常交给 optimizer 更新参数，不手动改写参数。需要函数式的新结果时，优先使用非原地表达式，例如 `y = x + 1`。

## 11. 常见报错与排查清单

### `.grad` 是 `None`

逐项检查：

1. 目标张量是否 `requires_grad=True`，dtype 是否为浮点或复数。
2. 计算是否在 `no_grad()` / `inference_mode()` 内，或张量是否已 `detach()`。
3. 该张量是否为非叶子张量；非叶子的 `.grad` 默认不保留，可调用 `retain_grad()`。
4. 是否已经调用 `.backward()`，且目标确实依赖该张量。
5. 是否在读取前由 `zero_grad()` 清除了梯度。

### `element 0 of tensors does not require grad`

通常是 loss 不在 autograd 图里。检查模型输出、损失是否被 `.detach()`、`.item()`、NumPy 转换或 `no_grad()` 断开。`.item()` 只适合记录/打印标量，不能再把 Python 数值当作可微损失。

### 非标量不能直接 `.backward()`

输出不是单个元素时，传入同形状的 `gradient`，或先用 `.sum()` / `.mean()` 得到符合目标含义的标量。

### 第二次反向传播时报计算图已释放

通常是同一图被反向传播两次。检查是否真的需要复用图；必要时在第一次调用中使用 `retain_graph=True`，或调整代码让每次迭代重新做前向计算。不要无差别地长期保留图。

### 梯度比预期大

检查是否遗漏 `optimizer.zero_grad()`、是否重复 backward，以及损失是否使用 `sum` 而非 `mean`。

### 需要梯度的叶子张量不能原地修改

将赋值/更新放到 `with torch.no_grad():` 中，或改成创建新张量的非原地运算。对视图、切片和共享存储也要特别谨慎。

### 显存不断增长

不要把带计算图的 loss 或输出长期存进列表。记录数值时用 `loss.item()`；保存张量供日志或可视化时用 `tensor.detach()`，并按需要移到 CPU。还应检查是否不必要地设置了 `retain_graph=True` 或 `create_graph=True`。

## 12. 梯度检查：用有限差分做直觉验证

对简单函数，可用中心差分近似导数：

\[
f'(x) \approx \frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}
\]

```python
def f(x):
    return x ** 3

x = torch.tensor(2.0, requires_grad=True)
f(x).backward()
autograd_grad = x.grad.item()

eps = 1e-4
finite_diff = (f(torch.tensor(2.0 + eps)) - f(torch.tensor(2.0 - eps))) / (2 * eps)
print(autograd_grad)       # 12
print(finite_diff.item())  # 约等于 12
```

有限差分受浮点舍入和 `eps` 选择影响，仅适合验证直觉。编写自定义 autograd 函数时，正式检查可用 `torch.autograd.gradcheck`，通常使用 `float64` 以提高数值精度。

## 13. 设备、精度和梯度注意事项

- 模型参数、输入和目标参与同一运算时，通常需要位于兼容的设备上，并满足算子要求的 dtype。
- 梯度 dtype 通常与对应参数的 dtype 一致；半精度训练需要注意梯度下溢，混合精度场景可使用 `torch.autocast` 和 `torch.amp.GradScaler`（具体支持取决于设备与 PyTorch 版本）。
- GPU 运算和部分算子可能存在非确定性；固定随机种子不保证所有环境下逐位一致。
- 检查数值异常可用 `torch.isfinite(loss)`、`torch.isfinite(parameter.grad).all()`；梯度为 NaN/Inf 时，沿输入、激活、损失和学习率逐段排查。

## 14. 自定义 `autograd.Function`（进阶）

多数情况下应组合 PyTorch 已有张量运算，让 autograd 自动求导。只有当实现特殊前向/反向、与外部算子交互等确有需要时，才自定义 `torch.autograd.Function`：

```python
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x

x = torch.tensor(3.0, requires_grad=True)
y = Square.apply(x)
y.backward()
print(x.grad)  # 6
```

`backward` 返回值的个数和顺序必须与 `forward` 输入对应；不可导输入的位置返回 `None`。反向函数要接收上游梯度 `grad_output` 并继续传递链式法则。自定义实现应使用 `gradcheck` 验证，并考虑高阶梯度、设备、dtype 和保存张量的内存成本。

## 15. 端到端小练习：拟合直线

下面用 autograd 和 SGD 从数据中学习 `y = 2x + 1`：

```python
torch.manual_seed(0)
x = torch.linspace(-2, 2, 100).unsqueeze(1)
target = 2 * x + 1

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    loss = criterion(prediction, target)
    loss.backward()
    optimizer.step()

print("weight:", model.weight.item())  # 接近 2
print("bias:", model.bias.item())      # 接近 1
print("loss:", loss.item())
```

建议逐步观察：

```python
print(prediction.requires_grad)  # True
print(loss.grad_fn)              # 非 None
print(model.weight.grad)         # backward 后存在
```

尝试删除 `zero_grad`，观察梯度累积；将 `mean()` 改为 `sum()`，观察梯度尺度；在前向外包 `torch.no_grad()`，观察 loss 不再有可反向传播的图。

## 16. 学习检查清单

- 能从 `requires_grad=True` 张量出发，预测哪些运算结果会有 `grad_fn`。
- 能区分 `.grad`（数值）、`grad_fn`（反向节点）和计算图。
- 明白默认 `.backward()` 用于标量；非标量需要 `gradient` 或先做有意义的归约。
- 明白梯度默认累加，并能写出 `zero_grad → forward → loss → backward → step`。
- 能解释叶子与非叶子的 `.grad` 行为，知道何时需要 `retain_grad()`。
- 能区分 `model.eval()` 与 `no_grad()` / `inference_mode()`。
- 能解释 `detach()`、`retain_graph=True` 和 `create_graph=True` 解决的是不同问题。
- 遇到梯度为 `None`、重复反传、原地修改错误或显存增长时，能从计算图和梯度生命周期排查。

## 17. 推荐学习顺序与官方资料

1. 先运行第 2、4 节的小例子，手算导数并与 `.grad` 对照。
2. 阅读第 5、6 节，完整运行线性拟合，观察每个训练步骤的状态变化。
3. 学习第 7、8 节，区分训练、评估、推理及图分离的边界。
4. 熟悉第 9 至 14 节的高级接口，遇到实际需求时再使用。

- [PyTorch Autograd 机制](https://docs.pytorch.org/docs/stable/autograd.html)
- [Autograd: 自动求导](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [PyTorch 神经网络基础教程](https://pytorch.org/tutorials/beginner/basics/intro.html)

版本提示：autograd 的核心概念长期稳定，但具体 API 参数、混合精度推荐接口和算子支持可能随 PyTorch 版本变化；查阅文档时应优先选择与本机安装版本匹配的官方资料。
