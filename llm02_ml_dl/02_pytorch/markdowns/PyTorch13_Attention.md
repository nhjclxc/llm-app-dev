# Attention 注意力：从 Q、K、V 到手写 Self-Attention、Multi-Head 与 Transformer

注意力机制（Attention）让模型在处理一个位置时，动态决定应该从哪些位置读取信息、每个位置应读取多少。它最早用于 Seq2Seq 翻译中解决固定长度上下文向量的信息瓶颈，后来成为 Transformer、BERT、GPT、视觉 Transformer（ViT）和多模态模型的核心组件。

本文从“加权平均”这一最小直觉出发，逐步推导 Query、Key、Value（Q/K/V）、Scaled Dot-Product Attention、自注意力、多头注意力、交叉注意力、Padding Mask、Causal Mask，并使用 PyTorch 从零手写一个可训练的 Multi-Head Self-Attention。示例默认适用于 PyTorch 2.x：

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

## 1. 学习路线与先修知识

| 阶段 | 要回答的问题 | 本文内容 |
| --- | --- | --- |
| 直觉 | 为什么模型需要“选择性读取”信息？ | 2-4 |
| 基础 | Q/K/V、打分、softmax 与加权求和是什么？ | 5-9 |
| 核心公式 | Scaled Dot-Product Attention 为什么要缩放？ | 10-12 |
| 序列内部关系 | Self-Attention 如何让 token 彼此通信？ | 13-15 |
| Transformer 核心 | Multi-Head Attention 如何并行关注不同关系？ | 16-19 |
| 多序列与生成 | Cross-Attention、Padding Mask、Causal Mask 如何工作？ | 20-24 |
| 工程实践 | 内置 API、完整 Block、训练与排错 | 25-31 |
| 深入理解 | 复杂度、位置编码、变体与练习路线 | 32-36 |

建议先掌握：Tensor 基本操作、矩阵乘法、`nn.Linear`、softmax、`nn.Module`、反向传播和 `DataLoader`。本文统一采用 batch-first 记法：

```text
[B, L, D]
B: batch size，批大小
L: sequence length，序列长度
D: embedding dimension / d_model，特征维度
```

## 2. 从“加权平均”理解注意力

假设你有若干条候选信息 `v_1, v_2, ..., v_L`。普通平均会给每条信息相同权重：

$$
\text{mean}(v)=\frac{1}{L}\sum_{j=1}^L v_j
$$

注意力使用可变权重 `alpha_j`：

$$
\text{Attention}(v)=\sum_{j=1}^L \alpha_jv_j,
\qquad \alpha_j\ge0,
\qquad\sum_{j=1}^L\alpha_j=1
$$

如果当前任务更需要第 2 条和第 4 条信息，模型可以学习让它们的权重更大。最小代码例子：

```python
values = torch.tensor([
    [1.0, 0.0],
    [0.0, 2.0],
    [3.0, 1.0],
])  # [L=3, D_v=2]

weights = torch.tensor([0.2, 0.7, 0.1])  # 所有权重非负，且和为 1
context = weights @ values

print(context)  # tensor([0.5, 1.5])
```

这里真正需要学习的是“当前应如何产生 `weights`”。Q/K/V 正是生成这些权重的机制。

## 3. 为什么 RNN 的固定上下文会成为瓶颈

早期机器翻译常使用 Encoder-Decoder RNN：编码器把整个源句压缩为一个最终隐藏状态 `h_T`，解码器仅依赖这个固定向量生成目标句。

```text
源句 token_1 ... token_T
          ↓ Encoder RNN
固定上下文向量 h_T
          ↓ Decoder RNN
目标句 token_1 ... token_U
```

短句尚可，但长句的全部细节被挤入一个固定长度向量，早期词语很容易丢失。注意力让解码器在生成每个目标 token 时，都能重新查看源句所有位置，并为源位置分配不同权重。

下例用人为分数展示不同解码步骤可读取不同源位置：

```python
encoder_states = torch.randn(1, 5, 8)  # [B=1, source_length=5, D=8]

# 假设生成目标第一个词更关注源位置 0；生成第二个词更关注源位置 3
scores_step_1 = torch.tensor([[4.0, 1.0, 0.0, -1.0, -2.0]])
scores_step_2 = torch.tensor([[-1.0, 0.0, 1.0, 4.0, -2.0]])

weights_1 = scores_step_1.softmax(dim=-1)
weights_2 = scores_step_2.softmax(dim=-1)
context_1 = weights_1.unsqueeze(1) @ encoder_states  # [1, 1, 8]
context_2 = weights_2.unsqueeze(1) @ encoder_states  # [1, 1, 8]

print(weights_1)
print(weights_2)
```

## 4. Attention 的三种角色：Query、Key、Value

注意力可类比为一次可检索的查询：

- **Query（Q，查询）**：当前位置“我在寻找什么？”
- **Key（K，键）**：每个候选位置“我能被怎样匹配？”
- **Value（V，值）**：候选位置“被选中后，我提供什么信息？”

流程如下：

```text
Query q 与所有 Key k_j 比较
             ↓
得到每个位置分数 score_j
             ↓ softmax
注意力权重 alpha_j
             ↓
用 alpha_j 对 Value v_j 加权求和
```

数学形式：

$$
s_j=\operatorname{score}(q,k_j),\qquad
\alpha_j=\operatorname{softmax}(s)_j,\qquad
\operatorname{Attention}(q,K,V)=\sum_j\alpha_jv_j
$$

Q、K、V 不一定来自不同数据；它们描述的是张量在本次注意力计算中的角色。最小单 Query 例子：

```python
q = torch.tensor([1.0, 0.0])
keys = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
])
values = torch.tensor([
    [10.0],
    [20.0],
    [30.0],
])

scores = q @ keys.T              # [3]
weights = torch.softmax(scores, dim=-1)
output = weights @ values        # [1]
print(scores, weights, output)
```

由于 `q` 与第一个 key 的点积最大，第一个 value 会有相对更大的贡献。

## 5. 点积为什么能作为相似度分数

最常用的打分函数是点积：

$$
\operatorname{score}(q,k)=q^Tk
$$

若 `q` 和 `k` 方向相近，点积较大；若方向相反，点积可能为负。注意点积同时受向量长度影响，因此它不是严格的余弦相似度；但 Q、K 的线性投影参数会在训练中适应任务。

```python
q = F.normalize(torch.tensor([1.0, 2.0]), dim=0)
k_similar = F.normalize(torch.tensor([1.0, 2.1]), dim=0)
k_different = F.normalize(torch.tensor([-2.0, 1.0]), dim=0)

print("similar:", torch.dot(q, k_similar).item())
print("different:", torch.dot(q, k_different).item())
```

也存在加性注意力（Bahdanau Attention）等评分方式：

$$
\operatorname{score}(q,k)=w^T\tanh(W_q q + W_k k)
$$

以下是简化代码。现代 Transformer 通常采用更高效、易并行的缩放点积注意力：

```python
class AdditiveScore(nn.Module):
    def __init__(self, q_dim, k_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(k_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, q, keys):
        # q: [B, Dq], keys: [B, Lk, Dk]
        h = torch.tanh(self.q_proj(q).unsqueeze(1) + self.k_proj(keys))
        return self.score(h).squeeze(-1)  # [B, Lk]
```

## 6. Softmax：把任意分数变成注意力权重

点积结果可正可负，且不保证和为 1。softmax 将它变成概率分布：

$$
\operatorname{softmax}(s_i)=\frac{e^{s_i}}{\sum_j e^{s_j}}
$$

```python
scores = torch.tensor([2.0, 1.0, -1.0])
weights = torch.softmax(scores, dim=-1)

print(weights)
print(weights.sum())  # tensor(1.)
```

在 batch 注意力中，softmax 必须沿“key 位置”维度做归一化。对于分数 `[B, Lq, Lk]`，使用 `dim=-1`：

```python
scores = torch.randn(2, 3, 5)  # B=2，3个 query，各自匹配 5 个 key
weights = torch.softmax(scores, dim=-1)
print(weights.shape)             # [2, 3, 5]
print(weights[0, 0].sum())       # 1：一个 query 对全部 key 的权重和
```

错误示例是沿 query 维 `dim=1` 做 softmax；那会让不同 query 相互竞争 key，而不符合“每个 query 自己选择读取位置”的含义。

## 7. 先手写单头注意力：无 batch 版本

设有 `Lq` 个 query、`Lk` 个 key/value：

```python
def attention_no_batch(q, k, v):
    """
    q: [Lq, Dk]
    k: [Lk, Dk]
    v: [Lk, Dv]
    returns:
        output: [Lq, Dv]
        weights: [Lq, Lk]
    """
    scores = q @ k.T
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights

q = torch.randn(2, 4)
k = torch.randn(3, 4)
v = torch.randn(3, 6)
output, weights = attention_no_batch(q, k, v)
print(output.shape)   # [2, 6]
print(weights.shape)  # [2, 3]
```

矩阵乘法维度应始终检查：

```text
[Lq, Dk] @ [Dk, Lk] = [Lq, Lk]    # score
[Lq, Lk] @ [Lk, Dv] = [Lq, Dv]    # output
```

## 8. Batch Attention 的 shape 推导

实际训练通常有 batch 维度。令：

```text
Q: [B, Lq, Dk]
K: [B, Lk, Dk]
V: [B, Lk, Dv]
```

使用 `K.transpose(-2, -1)` 后：

```text
K^T: [B, Dk, Lk]
Q @ K^T: [B, Lq, Lk]
weights @ V: [B, Lq, Dv]
```

对应代码：

```python
def batch_dot_product_attention(q, k, v):
    scores = q @ k.transpose(-2, -1)  # [B, Lq, Lk]
    weights = scores.softmax(dim=-1)
    output = weights @ v              # [B, Lq, Dv]
    return output, weights

q = torch.randn(2, 4, 8)
k = torch.randn(2, 6, 8)
v = torch.randn(2, 6, 10)
output, weights = batch_dot_product_attention(q, k, v)
print(output.shape)   # [2, 4, 10]
print(weights.shape)  # [2, 4, 6]
```

`Lq` 和 `Lk` 可以不同，这是交叉注意力的基础；只要求 Q 和 K 的最后一维相等，以便计算点积。

## 9. Q、K、V 通常由可学习线性投影得到

输入 token embedding 通常记作 `X`，shape `[B, L, D_model]`。注意力层使用三组独立参数投影出 Q/K/V：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

不要把 Q/K/V 理解为三份人工准备的数据；它们通常来自同一个或不同输入的不同线性变换：

```python
batch, length, d_model, d_k, d_v = 2, 5, 16, 8, 12
x = torch.randn(batch, length, d_model)

q_proj = nn.Linear(d_model, d_k, bias=False)
k_proj = nn.Linear(d_model, d_k, bias=False)
v_proj = nn.Linear(d_model, d_v, bias=False)

q = q_proj(x)
k = k_proj(x)
v = v_proj(x)
print(q.shape, k.shape, v.shape)  # [2, 5, 8], [2, 5, 8], [2, 5, 12]
```

即使输入相同，三组参数不同，模型也可以让某个 token 用一种特征“发起查询”、另一种特征“被匹配”、再用第三种特征“提供内容”。

## 10. Scaled Dot-Product Attention 的完整公式

Transformer 使用缩放点积注意力：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 `d_k` 是每个 query/key 向量的维度。计算步骤：

```text
Q @ K.transpose(-2, -1)       -> scores       [B, Lq, Lk]
scores / sqrt(d_k)             -> scaled score [B, Lq, Lk]
softmax(scaled score, -1)      -> weights      [B, Lq, Lk]
weights @ V                    -> output       [B, Lq, Dv]
```

函数实现：

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q:    [B, Lq, Dk]
    k:    [B, Lk, Dk]
    v:    [B, Lk, Dv]
    mask: 可广播到 [B, Lq, Lk] 的 bool Tensor；True 表示允许关注
    """
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights

q = torch.randn(2, 3, 8)
k = torch.randn(2, 4, 8)
v = torch.randn(2, 4, 6)
output, weights = scaled_dot_product_attention(q, k, v)
print(output.shape, weights.shape)  # [2, 3, 6]，[2, 3, 4]
```

## 11. 为什么要除以 `sqrt(d_k)`

假设 Q、K 各分量均值为 0、方差为 1，长度为 `d_k` 的点积是多项乘积之和，其方差会随 `d_k` 增大。维度较大时，score 值更极端，softmax 会接近 one-hot，梯度容易变得很小。

简单演示：

```python
for d_k in [4, 64, 512]:
    q = torch.randn(10_000, d_k)
    k = torch.randn(10_000, d_k)
    raw_scores = (q * k).sum(dim=-1)
    scaled_scores = raw_scores / math.sqrt(d_k)
    print(
        f"d_k={d_k:>3}, raw std={raw_scores.std():.2f}, "
        f"scaled std={scaled_scores.std():.2f}"
    )
```

缩放后 score 的量级更稳定，softmax 不会过早饱和。这里的缩放只针对 QK 分数，不针对 Value，也不改变输出维度。

## 12. 数值稳定性与 mask 后的 softmax

深度学习库的 `torch.softmax` 会采用稳定实现；不要手写 `torch.exp(scores) / torch.exp(scores).sum()`，大分数可能溢出。

```python
scores = torch.tensor([1000.0, 999.0, 998.0])
print(torch.softmax(scores, dim=-1))  # 安全且有限
```

mask 时，禁止位置的 score 应在 softmax 前改为极小值。对于浮点类型，`-inf` 最清晰：

```python
scores = torch.tensor([[2.0, 1.0, 0.0]])
allowed = torch.tensor([[True, True, False]])
masked_scores = scores.masked_fill(~allowed, float("-inf"))
weights = masked_scores.softmax(dim=-1)
print(weights)  # 最后位置权重为 0
```

每一行至少必须保留一个可关注位置；若整行全是 `-inf`，softmax 会得到 NaN：

```python
all_blocked = torch.tensor([[False, False]])
# 不要执行：torch.tensor([[1.0, 2.0]]).masked_fill(~all_blocked, -inf).softmax(-1)
```

## 13. Self-Attention：序列内部的全局通信

自注意力（Self-Attention）中，Q、K、V 都由同一个序列 `X` 投影得到：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

每个 token 都可基于当前内容查询整条序列。例如句子“这只动物没有过马路，因为它太累了”中，`它` 可以给前面不同 token 分配不同注意力，学习其指代关系。

```python
x = torch.randn(2, 5, 16)  # [B, L, D_model]

q_proj = nn.Linear(16, 16)
k_proj = nn.Linear(16, 16)
v_proj = nn.Linear(16, 16)

q, k, v = q_proj(x), k_proj(x), v_proj(x)
context, weights = scaled_dot_product_attention(q, k, v)
print(context.shape)  # [2, 5, 16]
print(weights.shape)  # [2, 5, 5]
```

`weights[b, i, j]` 表示 batch `b` 中第 `i` 个 query token 从第 `j` 个 token 读取信息的权重。注意：行对应 query，列对应 key/value。

## 14. Self-Attention 不等于“只关注自己”

Self 的含义是 Q/K/V 来自同一序列，不是说一个 token 只能关注自身。未加 mask 时，它可以关注所有位置，包括自己。

```python
weights = torch.tensor([
    [0.60, 0.25, 0.15],
    [0.20, 0.50, 0.30],
    [0.10, 0.40, 0.50],
])  # [Lq=3, Lk=3]

for query_position in range(3):
    print(
        f"query={query_position}, "
        f"most-attended key={weights[query_position].argmax().item()}"
    )
```

普通双向 self-attention（如 BERT encoder）允许 token 同时读取左侧和右侧上下文；因果 self-attention（如 GPT decoder）会用下三角 mask 禁止读取未来 token，详见第 23 节。

## 15. 一个可训练的单头 Self-Attention 模块

下面封装单头自注意力。输入和输出均为 `[B, L, D_model]`，适合放入更大模型中：

```python
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_head=None, dropout=0.0):
        super().__init__()
        d_head = d_model if d_head is None else d_head
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.out_proj = nn.Linear(d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, L, D_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        d_k = q.size(-1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = weights @ v
        return self.out_proj(context), weights

attention = SingleHeadSelfAttention(d_model=16, d_head=8, dropout=0.1)
x = torch.randn(4, 7, 16)
output, weights = attention(x)
print(output.shape, weights.shape)  # [4, 7, 16]，[4, 7, 7]
```

attention dropout 是对 softmax 后权重的随机失活，训练时用于正则化；`eval()` 模式下自动关闭。

## 16. 为什么单头注意力不够：Multi-Head Attention

单头 attention 在一个投影子空间中产生一份注意力分布。多头注意力（Multi-Head Attention，MHA）用 `h` 组不同的投影并行计算，让不同 head 可以学习不同关系，例如：

- 一个 head 关注邻近词；
- 一个 head 关注主谓关系；
- 一个 head 关注长距离指代；
- 一个 head 关注标点或句子边界。

第 `i` 个 head：

$$
\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

拼接全部 head 后再投影：

$$
\operatorname{MHA}(Q,K,V)=\operatorname{Concat}(\operatorname{head}_1,...,\operatorname{head}_h)W^O
$$

通常 `d_model` 能被 `num_heads` 整除，令：

$$
d_{head}=d_{model}/h
$$

```python
d_model = 64
num_heads = 8
assert d_model % num_heads == 0
print(d_model // num_heads)  # 每个 head 的 d_head=8
```

## 17. 多头注意力中的 reshape 与 transpose

线性投影后，Q 的常见 shape 是 `[B, L, D_model]`。为了并行计算所有头，需要变成 `[B, H, L, D_head]`：

```python
def split_heads(x, num_heads):
    """x: [B, L, D_model] -> [B, H, L, D_head]"""
    b, length, d_model = x.shape
    assert d_model % num_heads == 0
    d_head = d_model // num_heads
    return x.view(b, length, num_heads, d_head).transpose(1, 2)

def merge_heads(x):
    """x: [B, H, L, D_head] -> [B, L, H * D_head]"""
    b, heads, length, d_head = x.shape
    return x.transpose(1, 2).contiguous().view(b, length, heads * d_head)

x = torch.randn(2, 5, 16)
heads = split_heads(x, num_heads=4)
merged = merge_heads(heads)
print(heads.shape)   # [2, 4, 5, 4]
print(merged.shape)  # [2, 5, 16]
print(torch.allclose(x, merged))  # True
```

`transpose()` 后内存通常不连续，因此在 `view()` 前必须使用 `contiguous()`，或改用更能处理非连续张量的 `reshape()`。若省略，可能报错或产生不符合预期的结果。

## 18. 手写 Multi-Head Self-Attention

这是一个完整、可复用的手写模块。它支持 padding mask 和 causal mask；mask 约定是：`True` 表示允许关注，`False` 表示禁止关注。

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # [B, L, D] -> [B, H, L, Dh]
        b, length, _ = x.shape
        return x.view(b, length, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):
        # [B, H, L, Dh] -> [B, L, D]
        b, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, length, self.d_model)

    def forward(self, x, mask=None, need_weights=False):
        """
        x:    [B, L, D]
        mask: 可广播到 [B, H, L, L]，True=允许，False=屏蔽
        """
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        context = weights @ v
        output = self.out_proj(self._merge_heads(context))

        if need_weights:
            return output, weights
        return output

mha = MultiHeadSelfAttention(d_model=32, num_heads=4, dropout=0.1)
x = torch.randn(2, 6, 32)
output, weights = mha(x, need_weights=True)
print(output.shape)   # [2, 6, 32]
print(weights.shape)  # [2, 4, 6, 6]
```

这里每个 head 独立计算 `[L, L]` 注意力矩阵。最终 `weights` 的四个维度分别是 `[batch, head, query_position, key_position]`。

## 19. 验证手写模块的关键不变量

在大型模型中，shape 错误可能没有立刻报错，而是通过广播得到错误结果。先用断言检查：

```python
batch, length, d_model, heads = 3, 7, 24, 6
model = MultiHeadSelfAttention(d_model, heads)
x = torch.randn(batch, length, d_model)
y, weights = model(x, need_weights=True)

assert y.shape == x.shape
assert weights.shape == (batch, heads, length, length)
assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-5)
assert torch.isfinite(y).all()
print("shape and attention-weight checks passed")
```

训练时也应验证 Q/K/V 投影能接收梯度：

```python
loss = y.square().mean()
loss.backward()
print(model.q_proj.weight.grad.norm().item() > 0)  # 通常为 True
```

## 20. Cross-Attention：查询来自一条序列，信息来自另一条序列

交叉注意力（Cross-Attention）中：

- Q 来自 query sequence，长度为 `Lq`；
- K、V 来自 context / memory sequence，长度为 `Lk`；
- 输出与 query sequence 对齐，shape 是 `[B, Lq, D]`。

Seq2Seq 翻译的 decoder 常用 cross-attention：decoder 当前状态提出 Q，encoder 输出提供 K/V。

```text
encoder hidden states -> K, V
decoder hidden states -> Q
                         ↓
decoder 从源句所有位置读取上下文
```

最小例子：

```python
query_tokens = torch.randn(2, 3, 16)   # 目标序列/问题 token：[B, Lq=3, D]
memory_tokens = torch.randn(2, 5, 16)  # 源序列/图像特征：[B, Lk=5, D]

q_proj = nn.Linear(16, 16)
k_proj = nn.Linear(16, 16)
v_proj = nn.Linear(16, 16)

output, weights = scaled_dot_product_attention(
    q_proj(query_tokens),
    k_proj(memory_tokens),
    v_proj(memory_tokens),
)
print(output.shape, weights.shape)  # [2, 3, 16]，[2, 3, 5]
```

这里 `weights[b, i, j]` 表示第 `i` 个 query token 对第 `j` 个 memory token 的注意力。

## 21. 手写 Multi-Head Cross-Attention

与 self-attention 的区别只在输入：Q 用 `query` 投影，K/V 用 `context` 投影。

```python
class MultiHeadCrossAttention(MultiHeadSelfAttention):
    def forward(self, query, context, mask=None, need_weights=False):
        """
        query:   [B, Lq, D]
        context: [B, Lk, D]
        mask:    可广播到 [B, H, Lq, Lk]，True=允许
        """
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(context))
        v = self._split_heads(self.v_proj(context))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        context_out = weights @ v
        output = self.out_proj(self._merge_heads(context_out))
        return (output, weights) if need_weights else output

cross_attention = MultiHeadCrossAttention(d_model=32, num_heads=4)
query = torch.randn(2, 3, 32)
memory = torch.randn(2, 5, 32)
output, weights = cross_attention(query, memory, need_weights=True)
print(output.shape, weights.shape)  # [2, 3, 32]，[2, 4, 3, 5]
```

## 22. Mask 的统一约定

注意力 mask 不是额外特征，而是“哪些 query-key 连接合法”的规则。本文手写实现统一约定：

```text
mask == True  : 允许计算和关注
mask == False : 在 softmax 前将 score 置为 -inf，最终权重为 0
```

这种约定与 `masked_fill(~mask, -inf)` 自然匹配。但不同 PyTorch API 的 mask 约定可能不同，使用内置模块时必须查看其文档和参数说明。

一个一般 attention score mask 的 shape 是：

```text
[B, H, Lq, Lk]
```

实际可利用广播传入更小 shape：

| mask shape | 含义 |
| --- | --- |
| `[Lq, Lk]` | 所有样本、所有 head 共用同一规则，例如 causal mask |
| `[B, 1, 1, Lk]` | 每个样本有不同有效 key 长度，例如 padding mask |
| `[B, 1, Lq, Lk]` | 每个样本、每个 query 有不同规则 |
| `[B, H, Lq, Lk]` | 每个 head 独立规则，较少见 |

广播示例：

```python
scores = torch.randn(2, 4, 3, 5)        # [B=2, H=4, Lq=3, Lk=5]
key_allowed = torch.tensor([
    [True, True, True, False, False],
    [True, True, True, True, False],
])  # [B, Lk]

mask = key_allowed[:, None, None, :]    # [B, 1, 1, Lk]
masked_scores = scores.masked_fill(~mask, float("-inf"))
print(masked_scores.shape)
```

## 23. Padding Mask：不要关注补齐 token

同一 batch 的句子通常长度不同，需要 padding 到同长度。PAD token 没有真实语义，不能作为 K/V 被其他 token 读取。

```python
token_ids = torch.tensor([
    [4, 8, 9, 2, 0, 0],  # 实际长度 4，0 是 PAD
    [3, 7, 5, 1, 6, 2],  # 实际长度 6
])
pad_id = 0
key_allowed = token_ids.ne(pad_id)  # [B, L]，非 PAD 为 True
print(key_allowed)
```

将其变为可用于多头 self-attention 的 mask：

```python
padding_mask = key_allowed[:, None, None, :]  # [B, 1, 1, L]

attention = MultiHeadSelfAttention(d_model=16, num_heads=4)
x = torch.randn(2, 6, 16)
output, weights = attention(x, mask=padding_mask, need_weights=True)

print(weights[0, 0, 0, 4:])  # tensor([0., 0.])：第一个样本不能读 PAD 位置
```

关键点：这只禁止**其他位置读取 padding 作为 key/value**。若后续还要在每个位置输出 token 预测，通常也应在 loss 中忽略 PAD 位置，例如 `CrossEntropyLoss(ignore_index=pad_id)`。

## 24. Causal Mask / Look-Ahead Mask：禁止读取未来

自回归语言模型在预测位置 `t` 时只能看到 `0...t`，不能偷看未来 token。因果 mask 是下三角矩阵：

```text
      key:  0 1 2 3
query 0:   T F F F
query 1:   T T F F
query 2:   T T T F
query 3:   T T T T
```

构建和使用：

```python
def causal_mask(length, device=None):
    # [L, L]，下三角为 True
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()

mask = causal_mask(4)
print(mask)

attention = MultiHeadSelfAttention(d_model=8, num_heads=2)
x = torch.randn(1, 4, 8)
_, weights = attention(x, mask=mask, need_weights=True)
print(weights[0, 0])
```

第 0 行只能关注 key 0；第 2 行只能关注 key 0、1、2。`[L, L]` 会自动广播到 `[B, H, L, L]`。

把 padding 和 causal 规则组合时，两者都必须满足，使用逻辑与：

```python
token_ids = torch.tensor([[4, 8, 9, 2, 0, 0]])
padding = token_ids.ne(0)[:, None, None, :]  # [B, 1, 1, L]
causal = causal_mask(token_ids.size(1)).view(1, 1, 6, 6)
combined_mask = padding & causal
print(combined_mask.shape)  # [1, 1, 6, 6]
```

## 25. `nn.MultiheadAttention`：PyTorch 内置实现

PyTorch 内置 `nn.MultiheadAttention` 能减少手写 shape 错误。注意它默认 `batch_first=False`，此处显式设置 `True`：

```python
mha = nn.MultiheadAttention(
    embed_dim=32,
    num_heads=4,
    dropout=0.1,
    batch_first=True,
)

x = torch.randn(2, 5, 32)
output, weights = mha(
    query=x,
    key=x,
    value=x,
    need_weights=True,
    average_attn_weights=False,
)
print(output.shape)   # [2, 5, 32]
print(weights.shape)  # [2, 4, 5, 5]
```

内置模块的 `key_padding_mask` 与本文手写 mask 的布尔语义相反：**`True` 表示忽略/屏蔽该 key 位置**。例如：

```python
token_ids = torch.tensor([
    [4, 8, 2, 0, 0],
    [3, 7, 5, 1, 2],
])
key_padding_mask = token_ids.eq(0)  # True=PAD=禁止关注

x = torch.randn(2, 5, 32)
output, _ = mha(x, x, x, key_padding_mask=key_padding_mask)
print(output.shape)
```

这与本文自定义模块的 `True=允许` 不同，不能混用。`attn_mask` 对 bool mask 的语义也应按当前 PyTorch 文档确认；实际工程中推荐用小张量验证一个被屏蔽位置的权重确实为零。

## 26. 使用 `F.scaled_dot_product_attention`

PyTorch 2.x 提供底层函数 `torch.nn.functional.scaled_dot_product_attention`，在支持的硬件与形状下可能使用更高效的 fused/Flash Attention 内核。它期望 head 已拆分：

```python
q = torch.randn(2, 4, 5, 8)  # [B, H, Lq, Dh]
k = torch.randn(2, 4, 6, 8)  # [B, H, Lk, Dh]
v = torch.randn(2, 4, 6, 8)  # [B, H, Lk, Dv]

output = F.scaled_dot_product_attention(
    q, k, v,
    dropout_p=0.0,
    is_causal=False,
)
print(output.shape)  # [2, 4, 5, 8]
```

该函数中 boolean `attn_mask` 的语义是 **`True` 表示该位置参与 attention**，与本文手写模块一致：

```python
allowed = torch.ones(5, 6, dtype=torch.bool)
allowed[:, -1] = False
output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
```

传入 `is_causal=True` 时不应同时再传一个与之冲突的因果 `attn_mask`。训练时 `dropout_p` 应传训练用概率，评估时必须显式传 `0.0`；函数不像 `nn.Dropout` 那样自动读取模块的 train/eval 状态。

## 27. Attention + Residual + LayerNorm：一个 Transformer 子层

纯 attention 只负责混合不同 token 的信息。Transformer 常将其放入残差连接、LayerNorm 和 dropout 中：

$$
y=\operatorname{LayerNorm}(x+\operatorname{Dropout}(\operatorname{Attention}(x)))
$$

以下是 post-norm 风格的最小模块：

```python
class AttentionSublayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attended = self.attn(x, mask=mask)
        return self.norm(x + self.dropout(attended))

block = AttentionSublayer(d_model=32, num_heads=4)
x = torch.randn(2, 6, 32)
print(block(x).shape)  # [2, 6, 32]
```

残差要求 `x` 和 attention 输出 shape 相同，因此输出投影 `W^O` 的常见目的之一是回到 `D_model`。`LayerNorm(d_model)` 对每个 token 的最后一维独立归一化，不依赖 batch 统计量。

## 28. Feed-Forward Network：每个位置独立的非线性变换

Transformer block 中 attention 后通常接 Position-wise FFN：它对每个位置使用相同参数，但不在 token 之间混合信息；token 间通信由 attention 完成。

$$
\operatorname{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2
$$

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)  # [B, L, D] -> [B, L, D]

ffn = FeedForward(d_model=32, d_ff=128)
print(ffn(torch.randn(2, 6, 32)).shape)
```

完整的简化 Transformer encoder block：

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, mask=mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
```

## 29. Pre-Norm 与 Post-Norm

上一节是 post-norm：`LayerNorm(x + sublayer(x))`。现代深层 Transformer 经常使用 pre-norm：`x + sublayer(LayerNorm(x))`，通常优化更稳定。

```python
class PreNormEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

model = PreNormEncoderBlock(d_model=48, num_heads=6, d_ff=192)
print(model(torch.randn(2, 10, 48)).shape)
```

两者没有简单的绝对优劣；选择后要让训练设置、初始化、学习率调度与模型深度匹配。初学项目中选一种并保持一致即可。

## 30. 一个最小的 token 分类模型

下面将 token id 转为 embedding，加入位置编码，经过两层 attention block 后输出每个 token 的类别 logits。它展示 Attention 如何接入真实的训练模型：

```python
class TinyTokenClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=64, num_heads=4, max_len=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            PreNormEncoderBlock(d_model, num_heads, d_ff=4 * d_model)
            for _ in range(2)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids):
        # token_ids: [B, L]
        b, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        # 本文自定义模块：True 表示可读的 key 位置
        padding_mask = token_ids.ne(0)[:, None, None, :]
        for layer in self.layers:
            x = layer(x, mask=padding_mask)
        return self.classifier(x)  # [B, L, num_classes]

token_ids = torch.tensor([
    [5, 7, 9, 2, 0, 0],
    [4, 3, 8, 1, 6, 2],
])
model = TinyTokenClassifier(vocab_size=100, num_classes=4)
logits = model(token_ids)
print(logits.shape)  # [2, 6, 4]
```

token-level loss 必须忽略 padding label：

```python
labels = torch.tensor([
    [1, 2, 0, 3, -100, -100],
    [0, 1, 3, 2, 1, 0],
])
criterion = nn.CrossEntropyLoss(ignore_index=-100)
loss = criterion(logits.transpose(1, 2), labels)  # [B, C, L] 对 [B, L]
print(loss.item())
```

## 31. 手写 Attention 的完整训练冒烟测试

以下任务让模型学习“每个位置输出当前 token 是否等于序列第一个 token”。虽然简单，但可验证 embedding、attention、loss、梯度和 optimizer 全部连通。

```python
vocab_size = 20
batch_size, length = 64, 8
model = TinyTokenClassifier(vocab_size=vocab_size, num_classes=2, d_model=32, num_heads=4)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss()

for step in range(100):
    tokens = torch.randint(1, vocab_size, (batch_size, length), device=device)
    labels = tokens.eq(tokens[:, :1]).long()  # [B, L]，每位与第一个 token 比较
    logits = model(tokens)
    loss = criterion(logits.transpose(1, 2), labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if (step + 1) % 20 == 0:
        accuracy = logits.argmax(dim=-1).eq(labels).float().mean()
        print(f"step={step + 1}, loss={loss.item():.4f}, acc={accuracy.item():.3f}")
```

该例不能证明模型在自然语言任务上的能力，却能捕获常见工程错误，例如 mask shape 错误、attention 无梯度、logits 维度错误或类别 loss 使用错误。

## 32. Attention 的计算与显存复杂度

Self-Attention 的 score 矩阵大小是 `[B, H, L, L]`，因此序列长度 `L` 增大时，时间和显存成本约为二次增长：

$$
O(L^2D)
$$

代码直观展示：

```python
for length in [128, 512, 2048]:
    elements = 2 * 8 * length * length  # 假设 B=2、H=8 的 score 元素数
    memory_mb = elements * 4 / 1024**2  # float32
    print(f"L={length:>4}, score matrix roughly {memory_mb:.1f} MB")
```

这不是完整训练显存，只是 score 张量；还要考虑 Q/K/V、权重、激活和反向传播。长上下文模型会采用 Flash Attention、稀疏注意力、滑动窗口、线性注意力、KV cache 等优化。

## 33. 位置编码：Attention 本身不感知顺序

若没有位置信息，self-attention 对 token 排列是置换等变的：把 token 顺序重排，输出也只会以相同方式重排，模型无法仅从内容区分“先后”。因此 Transformer 要把 position information 加到 token embedding 中。

最简单的是可学习位置 embedding：

```python
token_embedding = nn.Embedding(1000, 32)
position_embedding = nn.Embedding(128, 32)

token_ids = torch.tensor([[5, 10, 3, 8]])
positions = torch.arange(token_ids.size(1)).unsqueeze(0)
x = token_embedding(token_ids) + position_embedding(positions)
print(x.shape)  # [1, 4, 32]
```

经典正弦位置编码：

$$
PE(pos,2i)=\sin(pos/10000^{2i/D}),\qquad
PE(pos,2i+1)=\cos(pos/10000^{2i/D})
$$

```python
def sinusoidal_position_encoding(max_len, d_model, device=None):
    position = torch.arange(max_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

pe = sinusoidal_position_encoding(max_len=10, d_model=16)
print(pe.shape)  # [10, 16]
```

现代模型也常使用 RoPE（Rotary Position Embedding）或相对位置偏置；其核心目的仍是让 attention score 感知相对/绝对位置关系。

## 34. Encoder、Decoder 和 Encoder-Decoder 中 Attention 的位置

| 架构 | Self-Attention 规则 | Cross-Attention | 典型模型/任务 |
| --- | --- | --- | --- |
| Encoder-only | 双向，可读全部非 PAD token | 无 | BERT，分类、编码、检索 |
| Decoder-only | 因果，只读当前位置及过去 | 无 | GPT，文本生成 |
| Encoder-Decoder | Encoder 双向；Decoder 因果 | Decoder 读取 Encoder 输出 | T5，翻译、摘要 |

一个 encoder-decoder decoder layer 的简化结构如下：

```python
class DecoderCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask=None, memory_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=causal_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, mask=memory_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

decoder_x = torch.randn(2, 4, 32)
encoder_memory = torch.randn(2, 6, 32)
block = DecoderCrossAttentionBlock(32, 4, 128)
output = block(decoder_x, encoder_memory, causal_mask=causal_mask(4))
print(output.shape)  # [2, 4, 32]
```

## 35. 常见错误与排错清单

| 症状 | 常见原因 | 检查/修复 |
| --- | --- | --- |
| `mat1 and mat2 shapes cannot be multiplied` | `d_model`、head reshape 或线性层输入维不匹配 | 打印 Q/K/V 的最后一维 |
| softmax 后权重不和为 1 | softmax 的 `dim` 错误 | 对 `[B,H,Lq,Lk]` 使用 `dim=-1` |
| loss 是 NaN | 某行被 mask 为全 `-inf`，或学习率过大 | 检查每行至少一个 `True` |
| 模型偷看未来、验证异常好 | 漏掉 causal mask 或 mask 方向反了 | 打印 `[L,L]` mask 和权重矩阵 |
| padding 影响输出 | 未 mask K/V 或 loss 未忽略 PAD | `key_allowed` + `ignore_index` |
| `view` 报连续性错误 | `transpose` 后直接 `view` | 使用 `.contiguous().view(...)` |
| 内置/手写结果不同 | mask bool 语义不同、dropout 未关闭 | 验证 API 约定，设置 `eval()` |
| 输出 shape 正确但学不动 | 没有位置编码、残差/归一化缺失、lr 不合理 | 先做第 31 节冒烟测试 |

调试 attention weight 的最小代码：

```python
model = MultiHeadSelfAttention(d_model=16, num_heads=2)
x = torch.randn(1, 4, 16)
mask = causal_mask(4)
_, weights = model(x, mask=mask, need_weights=True)

print(weights.shape)      # [1, 2, 4, 4]
print(weights[0, 0])      # 第一个样本、第一个 head 的完整注意力矩阵
print(weights[0, 0].triu(1).abs().max())  # 因果 mask 下应接近 0
```

## 36. 从零到一练习路线

1. **加权平均**：运行第 2 节，手工修改 weights，观察 context 如何变化。
2. **单 Query Q/K/V**：运行第 4 节，替换 query，观察权重最大的 key 如何变化。
3. **无 batch 点积注意力**：实现第 7 节函数，严格写出每个矩阵乘法 shape。
4. **缩放点积注意力**：实现第 10 节，比较是否除以 `sqrt(d_k)` 时 softmax 的尖锐程度。
5. **单头 Self-Attention**：完成第 15 节并检查 `[B, L, L]` 权重每行和为 1。
6. **多头重排**：独立实现 `split_heads`、`merge_heads`，确保来回张量完全一致。
7. **Causal Mask**：用第 24 节验证未来位置的权重都为 0。
8. **Padding Mask**：构造不同长度 batch，验证 PAD key 的权重都为 0。
9. **Cross-Attention**：让 query 长度和 memory 长度不同，检查权重为 `[B, H, Lq, Lk]`。
10. **Transformer Block**：完成第 28 或 29 节，运行第 31 节训练冒烟测试。
11. **阅读内置实现**：用 `nn.MultiheadAttention` 与手写版对照，尤其验证 mask 语义。

最终自测：

```python
def smoke_test_attention():
    model = MultiHeadSelfAttention(d_model=24, num_heads=3)
    x = torch.randn(2, 5, 24)
    mask = causal_mask(5)
    output, weights = model(x, mask=mask, need_weights=True)

    assert output.shape == x.shape
    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-5)
    assert weights[..., 0, 1:].abs().max() < 1e-6
    print("Attention smoke test passed")

smoke_test_attention()
```

## 37. 总结

注意力的核心不是“一个新层”，而是一种内容相关的信息路由机制：

```text
Query：我现在需要什么？
Key：每个位置能否匹配这个需求？
Value：若被选中，应提供什么信息？
```

最重要公式：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+\text{mask}\right)V
$$

实践中应始终确认：

- `Q @ K^T` 的最后一维是 key 位置，softmax 必须沿此维计算；
- mask 是在 softmax 前应用，且必须明确 `True/False` 的语义；
- self-attention 的 Q/K/V 同源，cross-attention 的 Q 与 K/V 异源；
- multi-head 的关键是 `[B,L,D] <-> [B,H,L,D_head]` 的正确变形；
- 注意力本身不编码位置，序列任务需要位置编码；
- 长序列的核心工程限制是 `O(L^2)` 的 score 矩阵。
