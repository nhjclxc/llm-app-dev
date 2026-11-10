# RNN 循环神经网络：从序列建模到 LSTM 和 GRU

循环神经网络（Recurrent Neural Network，RNN）用于处理有顺序关系的数据，例如文本、语音、时间序列、用户行为和传感器读数。与把每条样本视为独立向量的 MLP 不同，RNN 会在每个时间步维护隐藏状态（hidden state），让当前输出同时依赖当前输入和此前序列的信息。

本文从序列张量和手写循环开始，逐步学习 `nn.RNN`、`nn.LSTM`、`nn.GRU`、可变长度 batch、梯度消失、截断反向传播和完整训练流程。示例默认使用 PyTorch 2.x：

```python
import torch
from torch import nn
import torch.nn.functional as F
```

## 1. 学习路线与先修知识

| 阶段 | 要解决的问题 | 本文内容 |
| --- | --- | --- |
| 入门 | 序列、时间步、隐藏状态分别是什么？ | 2–5 |
| 基础实现 | 怎样使用手写循环和 `nn.RNN`？ | 6–9 |
| 训练原理 | RNN 怎样反向传播，为什么梯度会消失/爆炸？ | 10–12 |
| 长序列 | LSTM、GRU 怎样保存长期信息？ | 13–15 |
| 工程实践 | 变长序列、Embedding、双向 RNN、训练与评估 | 16–22 |
| 深入理解 | 状态管理、预测任务、排错与自测 | 23–26 |

建议先掌握 Tensor shape、`nn.Module`、`CrossEntropyLoss`、autograd、optimizer 和 DataLoader。RNN 的主线是：

```text
序列 x_1, x_2, ..., x_T
        ↓
每个时间步读取 x_t 与上一步状态 h_(t-1)
        ↓
更新为 h_t，并按任务产生一个或多个输出
```

## 2. 什么是序列建模

序列数据中的顺序通常有意义。句子中“我喜欢学习”和“学习喜欢我”由相同词组成，但含义不同；股票、温度和传感器读数的当前值也可能依赖历史趋势。序列建模的目标是在建模时保留这种依赖关系。

常见任务和输出形式：

| 任务 | 输入 | 输出 | 示例 |
| --- | --- | --- | --- |
| many-to-one | 一个序列 | 一个标签/数值 | 用评论序列判断情感；用过去 24 小时温度预测明天温度 |
| many-to-many（同步） | 一个序列 | 同长度序列 | 为每个词做词性标注 |
| next-step prediction | 前缀序列 | 下一个元素 | 语言模型预测下一个 token |
| encoder-decoder | 输入序列 | 另一序列 | 机器翻译、摘要 |

一个长度为 4、每步含 2 个特征的单样本时间序列可写作：

```python
x = torch.tensor([
    [0.1, 1.0],  # t=0
    [0.2, 0.8],  # t=1
    [0.3, 0.7],  # t=2
    [0.4, 0.5],  # t=3
])
print(x.shape)  # torch.Size([4, 2])，即 [T, D]
```

这里 `T` 是序列长度（时间步数），`D` 是每个时间步的输入特征数。`T` 不是 batch size；不同样本的 `T` 可以不同。

## 3. RNN 的输入输出形状

PyTorch 的循环层默认采用 sequence-first 格式：

```text
输入 input:  [T, N, D]
输出 output: [T, N, H * directions]
最终状态 h_n: [layers * directions, N, H]

T: sequence length，时间步数
N: batch size
D: input_size，每个时间步输入特征数
H: hidden_size，隐藏状态特征数
```

若设置 `batch_first=True`，输入和输出改为 `[N, T, D]` 与 `[N, T, H * directions]`；**最终隐藏状态 `h_n` 的 shape 不变**。初学时建议明确选择一种格式，以下多数 batch 示例使用 `batch_first=True`：

```python
batch = torch.randn(3, 5, 2)  # [N=3, T=5, D=2]
print(batch.shape)
```

`batch_first=True` 不改变“时间沿 T 维依次处理”的本质，只改变张量维度顺序。对于未分 batch 的单条序列，通常仍建议先 `unsqueeze(0)` 添加 batch 维。

## 4. 时间步与隐藏状态

最基本的 Elman RNN 在时间步 `t` 做如下计算：

$
h_t=\phi(W_{ih}x_t+W_{hh}h_{t-1}+b)
$

`x_t` 是当前输入，`h_(t-1)` 是上一步隐藏状态，`h_t` 是新隐藏状态，`phi` 常为 tanh 或 ReLU。隐藏状态是模型对“已读取前缀”的可学习摘要，不是人工规则，也不是直接保存全部历史数据。

用一个不含可训练参数的简化循环理解状态如何传递：

```python
sequence = torch.tensor([[1.0], [2.0], [3.0]])  # [T, D=1]
hidden = torch.zeros(1)                         # h_0

for t, x_t in enumerate(sequence):
    hidden = torch.tanh(x_t + hidden)
    print(f"t={t}, x_t={x_t.item():.1f}, h_t={hidden.item():.4f}")
```

每次循环的 `hidden` 都依赖此前值。若把 `hidden` 每步重新置零，模型就退化为独立处理每个时间步，无法建模跨时间依赖。

## 5. 输出来自哪里：全部时间步还是最后时间步

RNN 通常可获得每个时间步的输出，也可只使用最后有效时间步的信息：

```python
outputs = torch.randn(4, 6, 8)  # [N=4, T=6, H=8]

all_steps = outputs             # 用于序列标注，shape [4, 6, 8]
last_step = outputs[:, -1, :]   # 用于固定长度序列分类，shape [4, 8]
print(all_steps.shape, last_step.shape)
```

对等长、未填充的序列，`outputs[:, -1, :]` 可以表示最后时间步。对含 padding 的变长序列，最后一个位置可能是 padding，不能直接这样取；第 16 节会给出按真实长度取得最后有效输出的方法。

## 6. 从零实现一个 RNN Cell

`RNNCell` 只处理一个时间步。手写 cell 能帮助理解参数共享：同一个 `W_ih` 和 `W_hh` 在所有时间步重复使用，而不是每个时间步各有一套参数。

```python
class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t, h_prev):
        # x_t: [N, D]，h_prev: [N, H]
        return torch.tanh(self.input_to_hidden(x_t) + self.hidden_to_hidden(h_prev))

cell = SimpleRNNCell(input_size=2, hidden_size=4)
x_t = torch.randn(3, 2)
h_prev = torch.zeros(3, 4)
h_t = cell(x_t, h_prev)
print(h_t.shape)  # [3, 4]
```

用该 cell 展开一个完整序列：

```python
sequence = torch.randn(3, 5, 2)  # [N, T, D]
cell = SimpleRNNCell(2, 4)
hidden = torch.zeros(3, 4)
step_outputs = []

for t in range(sequence.size(1)):
    hidden = cell(sequence[:, t, :], hidden)
    step_outputs.append(hidden)

outputs = torch.stack(step_outputs, dim=1)
print(outputs.shape)  # [3, 5, 4]
print(hidden.shape)   # [3, 4]，最后时间步状态
```

实际项目通常使用 `nn.RNN`、`nn.LSTM` 或 `nn.GRU`，因为它们有高效实现并支持多层、双向和 packed sequence。手写循环在需要自定义状态更新规则时仍有价值。

## 7. `nn.RNN`：基础循环层

`nn.RNN` 将时间循环封装为一个层：

```python
rnn = nn.RNN(
    input_size=2,
    hidden_size=4,
    num_layers=1,
    nonlinearity="tanh",
    batch_first=True,
)

x = torch.randn(3, 5, 2)       # [N, T, D]
output, h_n = rnn(x)
print(output.shape)  # [3, 5, 4]，每个时间步最后层输出
print(h_n.shape)     # [1, 3, 4]，最后时间步的隐藏状态
```

重要参数：

| 参数 | 含义 |
| --- | --- |
| `input_size` | 每个时间步的特征数 D，不是序列长度 |
| `hidden_size` | 隐藏状态特征数 H |
| `num_layers` | 堆叠的 RNN 层数 |
| `nonlinearity` | `"tanh"` 或 `"relu"` |
| `batch_first` | 是否使用 `[N, T, D]` 输入格式 |
| `dropout` | 层与层之间的 dropout，仅当 `num_layers > 1` 时生效 |
| `bidirectional` | 是否建立正向和反向两个方向 |

初始隐藏状态默认全零，也可以显式传入。它的 shape 必须是 `[num_layers * directions, N, H]`：

```python
rnn = nn.RNN(2, 4, num_layers=2, batch_first=True)
x = torch.randn(3, 5, 2)
h0 = torch.zeros(2, 3, 4)  # [layers=2, N=3, H=4]
output, h_n = rnn(x, h0)
print(output.shape, h_n.shape)  # [3, 5, 4]， [2, 3, 4]
```

初始状态需与输入和模型处于同一设备、使用兼容 dtype。常用写法是 `x.new_zeros(...)`，可自动继承设备与 dtype。

## 8. 基于 `nn.RNN` 的序列分类

下例以每条序列的最后隐藏状态进行三分类。模型输出原始 logits，应直接交给 `CrossEntropyLoss`：

```python
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.rnn(x)
        last_layer_hidden = h_n[-1]       # [N, H]
        return self.head(last_layer_hidden)  # [N, C] logits

model = RNNClassifier(input_size=6, hidden_size=32, num_classes=3)
x = torch.randn(8, 12, 6)
labels = torch.randint(0, 3, (8,))
logits = model(x)
loss = nn.CrossEntropyLoss()(logits, labels)
print(logits.shape, loss.item())  # [8, 3]，标量 loss
```

`h_n[-1]` 代表最后一层、正向 RNN 的最终隐藏状态。它比 `output[:, -1]` 更直接表达“最后一步状态”，但对变长且未 packed 的输入，两者同样会受 padding 影响。

## 9. 多层 RNN 与 dropout

多层 RNN 把前一层每个时间步的输出序列作为下一层的输入序列：

```python
rnn = nn.RNN(
    input_size=10,
    hidden_size=16,
    num_layers=3,
    dropout=0.2,
    batch_first=True,
)
x = torch.randn(4, 7, 10)
output, h_n = rnn(x)
print(output.shape)  # [4, 7, 16]
print(h_n.shape)     # [3, 4, 16]，每层的最终状态
```

`dropout=0.2` 放在层与层之间，不会作用于最后一层输出；当 `num_layers=1` 时该参数不生效。不要假设它会在每个时间步、每个状态连接上做 dropout。需要更细粒度的正则化时，可在循环层前后显式加 `nn.Dropout`，但要先确认是否符合任务语义。

## 10. 通过时间的反向传播 BPTT

RNN 的参数在时间上共享，损失对某个参数的梯度来自所有时间步。反向传播会从后续时间步沿隐藏状态依赖逐步传回早期步骤，这称为 Backpropagation Through Time（BPTT）。

```text
x_1 -> h_1 -> h_2 -> h_3 -> loss
         ^      ^      ^
         └------共享同一组循环参数------┘
```

PyTorch 会在前向传播时建立计算图，`loss.backward()` 自动完成 BPTT：

```python
rnn = nn.RNN(1, 8, batch_first=True)
head = nn.Linear(8, 1)
x = torch.randn(2, 6, 1)
target = torch.randn(2, 1)

_, h_n = rnn(x)
prediction = head(h_n[-1])
loss = F.mse_loss(prediction, target)
loss.backward()

print(rnn.weight_hh_l0.grad.norm().item())
```

当序列很长时，计算图会占用更多内存，梯度还可能变得非常小或非常大。BPTT 是 RNN 能学习时序依赖的原因，也是其优化困难的来源。

## 11. 梯度消失与梯度爆炸

在 BPTT 中，梯度需要反复乘以循环权重和激活函数的局部导数。大量绝对值小于 1 的因子相乘会让早期时间步的梯度趋近于零，即梯度消失；大量大于 1 的因子相乘则可能让梯度急剧增大，造成梯度爆炸。

```python
# 训练后观察循环权重的梯度范数
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        print(name, parameter.grad.norm().item())
```

常见表现与应对：

| 现象 | 可能原因 | 常见处理 |
| --- | --- | --- |
| 长期依赖学不到、早期层梯度很小 | 梯度消失、tanh/sigmoid 饱和 | 用 LSTM/GRU、归一化、合适初始化、缩短 BPTT 段 |
| loss 突然变大或出现 NaN/Inf | 梯度爆炸、学习率过大、输入异常 | 降低学习率、检查数据、裁剪梯度 |
| 短序列能训练、长序列失败 | 时间依赖链过长 | LSTM/GRU、TBPTT、重新定义上下文窗口 |

梯度裁剪应在 `backward()` 后、`optimizer.step()` 前执行：

```python
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
print(float(grad_norm))  # 裁剪前的总梯度范数
```

裁剪限制的是更新所用梯度，不会修复标签错位、学习率明显不合理或数值溢出等根因。

## 12. 截断反向传播 TBPTT

对极长流式序列，若一直把隐藏状态连接到过去，计算图会无限增长。截断反向传播（Truncated BPTT，TBPTT）会在固定长度片段后保留数值状态、切断其到旧计算图的梯度连接：

```python
rnn = nn.GRU(input_size=8, hidden_size=32, batch_first=True)
hidden = None

for chunk in stream_chunks:  # 每段 shape: [N, chunk_length, 8]
    output, hidden = rnn(chunk, hidden)
    loss = loss_for_chunk(output)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
    optimizer.step()

    hidden = hidden.detach()  # 保留数值，但切断跨 chunk 的反向图
```

`detach()` 不会把隐藏状态清零，模型仍能在前向中使用前一片段的信息；它只限制梯度不再跨越片段边界。TBPTT 是内存与长期梯度传播能力之间的权衡，片段长度应由任务依赖范围和资源决定。

## 13. `nn.LSTM`：门控记忆

长短期记忆网络（Long Short-Term Memory，LSTM）在隐藏状态 `h_t` 之外维护细胞状态 `c_t`。它用输入门、遗忘门、输出门控制什么信息写入、保留和读取，从而缓解普通 RNN 的长期梯度问题。

简化公式：

$
f_t=\sigma(W_f[x_t,h_{t-1}]+b_f),\quad
i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)
$

$
\tilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c),\quad
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t
$

$
o_t=\sigma(W_o[x_t,h_{t-1}]+b_o),\quad
h_t=o_t\odot\tanh(c_t)
$

门值在 0 到 1 之间，逐元素决定信息流。不要把门理解为硬性的 if 语句；它们是从数据中学习到的连续权重。

使用 `nn.LSTM`：

```python
lstm = nn.LSTM(input_size=5, hidden_size=12, num_layers=2, batch_first=True)
x = torch.randn(4, 7, 5)
output, (h_n, c_n) = lstm(x)

print(output.shape)  # [4, 7, 12]
print(h_n.shape)     # [2, 4, 12]，每层最终隐藏状态
print(c_n.shape)     # [2, 4, 12]，每层最终细胞状态
```

显式初始状态要同时传入 `h0` 和 `c0`：

```python
h0 = torch.zeros(2, 4, 12)
c0 = torch.zeros(2, 4, 12)
output, (h_n, c_n) = lstm(x, (h0, c0))
```

LSTM 缓解而非彻底消除长期依赖问题；更长的序列仍需关注梯度、内存、数据质量和任务定义。

## 14. 基于 LSTM 的情感分类结构

文本 token ID 不能直接作为连续数字送入 RNN，应先经 `nn.Embedding` 转为可学习向量。下例输入已 padding 到相同长度，暂不处理变长问题：

```python
class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)  # [N, T] -> [N, T, E]
        _, (h_n, _) = self.lstm(embeddings)
        return self.head(h_n[-1])               # [N, C] logits

model = LSTMTextClassifier(vocab_size=5000, embedding_dim=64, hidden_size=128, num_classes=2)
token_ids = torch.randint(1, 5000, (8, 20))
labels = torch.randint(0, 2, (8,))
loss = F.cross_entropy(model(token_ids), labels)
print(loss.item())
```

`padding_idx=0` 使 embedding 的第 0 行不参与常规梯度更新，适合作为 padding token；但仅设置它不能让 LSTM 自动忽略 padding，第 16 节仍需处理长度或 packed sequence。

## 15. `nn.GRU`：更紧凑的门控循环层

门控循环单元（Gated Recurrent Unit，GRU）使用更新门和重置门控制状态，通常没有单独的细胞状态，因此参数比同尺寸 LSTM 少：

$
z_t=\sigma(W_z[x_t,h_{t-1}]),\qquad
r_t=\sigma(W_r[x_t,h_{t-1}])
$

$
\tilde h_t=\tanh(W_h[x_t,r_t\odot h_{t-1}]),\qquad
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t
$

使用方式与 `nn.RNN` 接近：

```python
gru = nn.GRU(input_size=5, hidden_size=12, num_layers=2, batch_first=True)
x = torch.randn(4, 7, 5)
output, h_n = gru(x)
print(output.shape)  # [4, 7, 12]
print(h_n.shape)     # [2, 4, 12]
```

RNN、LSTM、GRU 的主要差别：

| 层 | 状态 | 优点 | 局限 |
| --- | --- | --- | --- |
| `nn.RNN` | `h_t` | 简单、参数少、适合教学和短依赖 | 长依赖更易梯度消失/爆炸 |
| `nn.LSTM` | `h_t` 与 `c_t` | 门控完善，常用于较长依赖 | 参数更多、状态接口更复杂 |
| `nn.GRU` | `h_t` | 比 LSTM 紧凑，常是强基线 | 与 LSTM 没有普适的绝对优劣 |

实际选择应在验证集上比较效果、训练速度和参数规模。不要仅因为名称更新就假设 GRU 一定比 LSTM 更好。

## 16. 可变长度序列与 padding

一个 batch 需要组成规则 Tensor，因此不同长度序列常被补到相同长度。默认补零并不等于模型会自动忽略零；如果零是合法输入，或 RNN 继续处理 padding，会污染末尾状态和损失。

先用 `pad_sequence` 构造 batch：

```python
from torch.nn.utils.rnn import pad_sequence

sequences = [
    torch.tensor([4, 7, 2]),
    torch.tensor([5, 1]),
    torch.tensor([9, 3, 8, 6]),
]
lengths = torch.tensor([3, 2, 4])
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
print(padded)
print(padded.shape)  # [3, 4]
```

### 16.1 根据长度选择最后有效输出

若已得到未 packed 的 `outputs: [N, T, H]`，最后有效位置是 `lengths - 1`：

```python
outputs = torch.randn(3, 4, 8)
last_indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
last_valid = outputs.gather(dim=1, index=last_indices).squeeze(1)
print(last_valid.shape)  # [3, 8]
```

这避免错误地统一取 `outputs[:, -1]`。但 RNN 在该写法中仍计算了 padding 时间步；长短差距大时浪费计算，且双向 RNN 会受到 padding 影响。

### 16.2 `pack_padded_sequence`

PyTorch 可将补齐序列打包，让循环层跳过 padding：

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

embedding = nn.Embedding(20, 6, padding_idx=0)
lstm = nn.LSTM(6, 10, batch_first=True)

# pack 默认要求 lengths 从大到小；enforce_sorted=False 可保留当前顺序
embedded = embedding(padded)
packed = pack_padded_sequence(
    embedded,
    lengths.cpu(),       # lengths 必须在 CPU
    batch_first=True,
    enforce_sorted=False,
)
packed_output, (h_n, c_n) = lstm(packed)
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

print(output.shape)        # [3, 4, 10]
print(h_n[-1].shape)       # [3, 10]，每条序列的最后有效状态
print(output_lengths)      # tensor([3, 2, 4])
```

对单向 LSTM/GRU 序列分类，packed 后的 `h_n[-1]` 可直接作为最后有效状态。使用双向或多层模型时，需按第 17 节组合正确方向和层的状态。

## 17. 双向 RNN 与多层最终状态

双向 RNN 同时从左到右和从右到左读取序列，把两个方向输出拼接：

```python
gru = nn.GRU(6, 10, batch_first=True, bidirectional=True)
x = torch.randn(4, 5, 6)
output, h_n = gru(x)
print(output.shape)  # [4, 5, 20]，10 正向 + 10 反向
print(h_n.shape)     # [2, 4, 10]，directions=2

forward_last = h_n[-2]  # [N, H]
backward_last = h_n[-1] # [N, H]
representation = torch.cat([forward_last, backward_last], dim=1)
print(representation.shape)  # [4, 20]
```

若同时 `num_layers > 1` 和 `bidirectional=True`，`h_n` 第 0 维按“第 0 层正向、第 0 层反向、第 1 层正向、第 1 层反向”排列。最后一层的两个方向仍是 `h_n[-2]` 和 `h_n[-1]`。分类头输入维度应为 `hidden_size * 2`。

双向模型在每个位置可见未来上下文，因此适合完整句子分类、离线标注等任务；不能直接用于严格在线的未来预测，因为反向方向依赖未来输入。

## 18. 序列标注：每个时间步预测一个类别

词性标注、命名实体识别和逐帧分类属于 many-to-many 任务。循环层的每步输出经过线性层，产生每步 logits：

```python
class GRUTagger(nn.Module):
    def __init__(self, input_size, hidden_size, num_tags):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x):
        outputs, _ = self.gru(x)          # [N, T, 2H]
        return self.classifier(outputs)    # [N, T, num_tags]

model = GRUTagger(input_size=8, hidden_size=16, num_tags=5)
x = torch.randn(2, 4, 8)
tags = torch.randint(0, 5, (2, 4))
logits = model(x)

# CrossEntropyLoss 要求类别维在第 1 维，可转为 [N, C, T]
loss = F.cross_entropy(logits.transpose(1, 2), tags)
print(logits.shape, loss.item())  # [2, 4, 5]
```

若有 padding 标签，应在 `CrossEntropyLoss(ignore_index=pad_tag_id)` 中忽略它们，或用 mask 手动统计。只 mask 输入而不 mask loss，会让 padding 位置参与训练目标，造成错误梯度。

## 19. 下一个 token 预测

语言模型的训练目标通常是“输入前缀，预测下一个 token”。标签相对输入左移一位：

```python
tokens = torch.tensor([[2, 8, 4, 9, 3]])
inputs = tokens[:, :-1]   # [2, 8, 4, 9]
targets = tokens[:, 1:]   # [8, 4, 9, 3]

embedding = nn.Embedding(num_embeddings=20, embedding_dim=12)
gru = nn.GRU(12, 24, batch_first=True)
vocab_head = nn.Linear(24, 20)

outputs, _ = gru(embedding(inputs))
logits = vocab_head(outputs)  # [N, T-1, vocab_size]
loss = F.cross_entropy(logits.transpose(1, 2), targets)
print(loss.item())
```

训练时每个时间步都有一个词表分类损失。推理生成时，把已生成 token 逐步喂入模型，按 logits 的 softmax 分布选择下一个 token；贪心选择、温度采样、top-k、top-p 是不同解码策略。训练标签错一位是该任务最常见的逻辑错误之一。

## 20. 完整的序列分类训练循环

以下示例假设 DataLoader 每批返回 `features: [N, T, D]` 与类别标签 `labels: [N]`，所有序列已等长或已按前述方式处理。它体现训练和验证的关键边界：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNClassifier(input_size=6, hidden_size=32, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(10):
    model.train()
    train_loss_sum = 0.0
    train_count = 0

    for features, labels in train_loader:
        features = features.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        count = labels.size(0)
        train_loss_sum += loss.item() * count
        train_count += count

    model.eval()
    valid_correct = 0
    valid_count = 0
    with torch.inference_mode():
        for features, labels in valid_loader:
            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            logits = model(features)
            valid_correct += (logits.argmax(dim=1) == labels).sum().item()
            valid_count += labels.size(0)

    print(
        f"epoch={epoch + 1} "
        f"train_loss={train_loss_sum / train_count:.4f} "
        f"valid_acc={valid_correct / valid_count:.4f}"
    )
```

训练阶段使用 `model.train()`，验证阶段使用 `model.eval()` 和 `torch.inference_mode()`。`eval()` 切换 Dropout、BatchNorm 等模块的行为；`inference_mode()` 关闭梯度记录，二者作用不同，验证时通常都需要。

## 21. 状态是否跨 batch 保留

多数使用 DataLoader 的序列分类任务中，batch 中各样本互相独立，应该让 RNN 对每个 batch 使用默认零初始状态。不要无意地把一个无关 batch 的 `h_n` 传给下一个 batch。

只有连续流式数据确实被按正确顺序分块时，才考虑跨 batch 保留状态：

```python
hidden = None
for chunk, targets in ordered_stream_loader:
    chunk = chunk.to(device)
    output, hidden = gru(chunk, hidden)
    loss = loss_for_chunk(output, targets.to(device))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    hidden = hidden.detach()
```

此时必须保证同一 batch slot 在相邻 chunk 中仍对应同一条连续序列；如果 `shuffle=True`、最后 batch 尺寸变化或样本顺序重排，传递隐藏状态会把不同样本的信息混在一起。

## 22. 保存、加载和推理

保存模型时优先保存 state dict 与配置；重新构造相同结构后再加载：

```python
torch.save(
    {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    },
    "rnn_checkpoint.pt",
)

checkpoint = torch.load("rnn_checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
model.eval()
```

推理时，输入处理必须与训练一致：词表 ID 映射、未知词策略、padding 位置、长度处理、数值标准化参数和设备都属于模型契约的一部分。只保存权重、却不保存词表和预处理配置，通常无法可靠复现文本模型。

## 23. 常见错误与排查

| 问题 | 常见原因 | 检查方式 |
| --- | --- | --- |
| `input.size(-1) must be equal to input_size` | 把 T 当作 D，或 Embedding 维度不匹配 | 打印输入 `[N, T, D]` 与层的 `input_size` |
| 分类 loss 报 shape/dtype 错 | logits/target 格式不符 | 多分类应为 logits `[N, C]`、target `[N]` long |
| 变长序列效果异常 | 直接取 padding 后最后一步 | 使用 lengths、mask 或 packed sequence |
| 显存逐 batch 增长 | 跨 batch 传 hidden 但未 detach | 在 TBPTT 边界 `hidden = hidden.detach()` |
| 所有输出近似相同 | 标签错位、学习率、输入未标准化、梯度消失 | 先过拟合一个小 batch，检查 logits 与标签 |
| RNN 训练不稳定 | 长序列梯度爆炸 | 检查梯度范数、降低 lr、使用裁剪/LSTM/GRU |
| 验证指标波动很大 | 验证 loader 随机增强或随机状态未控 | 验证用确定性 transform、`shuffle=False` |
| 双向模型线上不可用 | 反向方向读取未来数据 | 改用单向模型或离线部署 |

调试层的输入输出 shape：

```python
model = nn.GRU(4, 8, num_layers=2, batch_first=True, bidirectional=True)
x = torch.randn(2, 3, 4)
output, h_n = model(x)
print("input:", x.shape)      # [2, 3, 4]
print("output:", output.shape)  # [2, 3, 16]
print("h_n:", h_n.shape)        # [4, 2, 8]
```

## 24. RNN 与 CNN、Transformer 的边界

RNN 按时间步顺序计算，天然适合流式输入和长度变化的序列，但训练难以并行化，远距离依赖也较困难。CNN 可在局部时间窗口并行提取模式；Transformer 的 self-attention 能直接建立任意位置关系并高度并行，但对长序列的注意力成本可能较高。

```python
# 同样形状的连续特征 [N, T, D]，不同模型对维度的要求不同
x = torch.randn(8, 20, 16)

rnn = nn.GRU(16, 32, batch_first=True)           # 直接使用 [N, T, D]
conv = nn.Conv1d(16, 32, kernel_size=3, padding=1) # 需转为 [N, D, T]
conv_output = conv(x.transpose(1, 2))

print(rnn(x)[0].shape)      # [8, 20, 32]
print(conv_output.shape)    # [8, 32, 20]
```

模型选择取决于序列长度、数据规模、实时性、解释和部署约束。RNN/LSTM/GRU 仍是小中型时序任务、流式预测和教学中重要的基线，并没有被 Transformer 完全替代。

## 25. 推荐练习顺序

1. 用第 6 节手写 cell，打印每个时间步的隐藏状态。
2. 把手写 cell 替换为 `nn.RNN`，核对 `output` 与 `h_n` 的 shape。
3. 用一个合成的 sin 波序列做 next-step 回归，比较 RNN、LSTM、GRU。
4. 用 token ID 与 `nn.Embedding` 做短文本二分类。
5. 给文本样本加入不同长度和 padding，分别尝试 gather 与 packed sequence。
6. 将单向 GRU 改为双向 GRU，调整分类头输入宽度并解释线上约束。
7. 故意移除 `hidden.detach()`，观察流式训练的内存/计算图问题，再恢复 TBPTT。

## 26. RNN 学习自测

1. `[N, T, D]` 中的 T 与 D 分别代表什么？`nn.RNN(input_size=...)` 应填哪个？
2. 普通 RNN 的 `h_t` 为什么能包含前文信息？它为什么又可能遗忘远处信息？
3. `nn.RNN` 输出的 `output` 与 `h_n` 分别包含什么？
4. `nn.LSTM` 为什么返回 `(h_n, c_n)`，而 `nn.GRU` 只返回 `h_n`？
5. 为什么对 padding 序列直接使用 `outputs[:, -1]` 可能错误？
6. 为什么跨 batch 传递隐藏状态时常要 `detach()`？
7. 双向 RNN 为什么不适合严格实时的下一个事件预测？
8. `CrossEntropyLoss` 的多分类 logits、target 应各是什么 shape 和 dtype？

建议先用手算与 shape 推理回答，再运行相应小例子验证。掌握序列维、状态维、层数维和方向维这四类维度，是正确使用 PyTorch 循环网络的基础。
