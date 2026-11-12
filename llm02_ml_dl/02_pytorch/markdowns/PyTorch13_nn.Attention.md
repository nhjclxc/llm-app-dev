# PyTorch 内置 Attention 组件：从 MultiheadAttention 到 FlexAttention

本文说明 PyTorch 2.x 的内置注意力组件：`nn.MultiheadAttention`、`F.scaled_dot_product_attention`（SDPA）、`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`、`nn.TransformerEncoder`、`nn.TransformerDecoder` 与 FlexAttention。

本文关注工程使用：输入输出 shape、mask 语义、组件组合、训练与性能边界。Q/K/V、缩放点积与手写 Attention 原理可配合阅读 `PyTorch13_Attention.md`。

```python
import math
import inspect
import torch
from torch import nn
import torch.nn.functional as F

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 1. 组件地图与选择原则

| 层级 | API | 适用场景 |
| --- | --- | --- |
| 模块级 | `nn.MultiheadAttention` | 标准 MHA，或需要查看 attention 权重 |
| 函数级 | `F.scaled_dot_product_attention` | 自定义 Q/K/V 投影且希望使用高效内核 |
| 子层级 | `TransformerEncoderLayer` / `TransformerDecoderLayer` | 标准 Transformer 子层 |
| 堆叠级 | `TransformerEncoder` / `TransformerDecoder` | 标准 encoder/decoder 网络 |
| 内核级 | `flex_attention` | 自定义 score、稀疏或分块 attention |

```text
标准 MHA                         -> nn.MultiheadAttention
自定义投影 + 高效 attention kernel -> F.scaled_dot_product_attention
标准 Transformer                 -> TransformerEncoder/Decoder Layer
特殊 mask 或稀疏 attention        -> FlexAttention
```

抽象越低，灵活性越大，调用者也越需要保证 shape、mask、dropout 与性能正确。

## 2. 统一 shape 与公式

本文优先设置 `batch_first=True`：

```text
[B, L, E]
B: batch size
L: sequence length
E: embedding dimension / d_model
H: num_heads
Dh: 每个 head 的维度，通常 E / H
```

注意力公式：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{D_h}}+M\right)V
$$

```python
q = torch.randn(2, 3, 8)   # [B, Lq, Dk]
k = torch.randn(2, 5, 8)   # [B, Lk, Dk]
v = torch.randn(2, 5, 12)  # [B, Lk, Dv]

scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
weights = scores.softmax(dim=-1)
output = weights @ v
print(output.shape)  # [2, 3, 12]
```

拆分 head 后，SDPA 常使用 `[B, H, L, Dh]`；完整 attention 权重常为 `[B, H, Lq, Lk]`。

## 3. `nn.MultiheadAttention`：标准多头注意力

`nn.MultiheadAttention` 封装 Q/K/V 线性投影、拆分 head、缩放点积 attention、合并 head 与输出投影：

```python
mha = nn.MultiheadAttention(
    embed_dim=32,
    num_heads=4,
    dropout=0.1,
    batch_first=True,
)

x = torch.randn(2, 5, 32)
output, weights = mha(x, x, x)
print(output.shape)   # [2, 5, 32]
print(weights.shape)  # 默认平均 head 后：[2, 5, 5]
```

Self-Attention 令 `query=key=value=x`。`embed_dim` 必须能被 `num_heads` 整除；这里每个 head 的维度是 `32 // 4 = 8`。

## 4. MHA 构造参数与 Cross-Attention

常见构造参数：

```python
nn.MultiheadAttention(
    embed_dim,
    num_heads,
    dropout=0.0,
    bias=True,
    kdim=None,
    vdim=None,
    batch_first=False,
)
```

| 参数 | 说明 |
| --- | --- |
| `embed_dim` | query 与输出的特征维 E |
| `num_heads` | head 数；E 必须能整除它 |
| `dropout` | attention 权重 dropout 概率 |
| `kdim` / `vdim` | key/value 输入维度，默认等于 `embed_dim` |
| `batch_first` | True 使用 `[B,L,E]`；默认 False 使用 `[L,B,E]` |

Cross-Attention 示例：

```python
mha = nn.MultiheadAttention(
    embed_dim=32, num_heads=4, kdim=24, vdim=40, batch_first=True
)
query = torch.randn(2, 3, 32)
key = torch.randn(2, 5, 24)
value = torch.randn(2, 5, 40)
output, weights = mha(query, key, value)

print(output.shape)   # [2, 3, 32]，长度跟 query 对齐
print(weights.shape)  # [2, 3, 5]
```

## 5. MHA forward、attention 权重与性能开关

核心调用：

```python
output, weights = mha(
    query, key, value,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False,
)
```

查看每个 head 的独立权重：

```python
x = torch.randn(2, 6, 32)
mha = nn.MultiheadAttention(32, 4, batch_first=True)
output, weights = mha(
    x, x, x,
    need_weights=True,
    average_attn_weights=False,
)
print(output.shape)   # [2, 6, 32]
print(weights.shape)  # [2, 4, 6, 6]
```

不需要权重时应设 `need_weights=False`，能减少返回/物化开销，且通常有更多优化机会：

```python
output, _ = mha(x, x, x, need_weights=False)
```

## 6. MHA 的 Padding Mask 与 Causal Mask

`nn.MultiheadAttention` 的 bool mask 约定：

- `key_padding_mask`: `[B,Lk]`，`True` 表示该 key 被忽略。
- `attn_mask`: 通常 `[Lq,Lk]`，`True` 表示禁止这一 query-key 连接。

Padding mask：

```python
pad_id = 0
token_ids = torch.tensor([[5, 9, 2, 0, 0], [4, 8, 7, 1, 2]])
x = torch.randn(2, 5, 32)
padding_mask = token_ids.eq(pad_id)  # True=PAD=忽略

mha = nn.MultiheadAttention(32, 4, batch_first=True)
_, weights = mha(
    x, x, x,
    key_padding_mask=padding_mask,
    need_weights=True,
    average_attn_weights=False,
)
print(weights[0, :, :, 3:].abs().max())  # 被忽略 key 的权重应为 0
```

Causal mask：

```python
L = 5
causal_block = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
# 上三角 True：禁止读取未来 token

_, weights = mha(
    x, x, x,
    attn_mask=causal_block,
    need_weights=True,
    average_attn_weights=False,
)
print(weights[0, 0].triu(1).abs().max())  # 未来权重为 0
```

float mask 会加到 score，屏蔽位置一般填 `-inf`。MHA bool mask 语义与 SDPA bool mask 相反，不能直接复用。

## 7. MHA 参数、梯度与 `batch_first`

默认 MHA 常将 Q/K/V 投影拼接存储：

```python
mha = nn.MultiheadAttention(32, 4, batch_first=True)
for name, parameter in mha.named_parameters():
    print(name, tuple(parameter.shape))

x = torch.randn(2, 5, 32)
y, _ = mha(x, x, x, need_weights=False)
y.square().mean().backward()
print(mha.in_proj_weight.grad.norm())
```

默认 `batch_first=False` 时，输入为 `[L,B,E]`：

```python
mha_seq_first = nn.MultiheadAttention(32, 4)
x_batch_first = torch.randn(2, 5, 32)
x_seq_first = x_batch_first.transpose(0, 1)
y, _ = mha_seq_first(x_seq_first, x_seq_first, x_seq_first)
print(y.shape)  # [5, 2, 32]
```

把 `[B,L,E]` 误喂给默认模块有时不报错，却会把 batch 与长度解释反，是高风险静默错误。

## 8. `F.scaled_dot_product_attention`（SDPA）

SDPA 是函数级组件：只完成 score、mask、softmax、dropout、乘 V。调用者负责 Q/K/V 投影、拆/合 head 和输出投影。

```text
query:  [B,H,Lq,Dh]
key:    [B,H,Lk,Dh]
value:  [B,H,Lk,Dv]
output: [B,H,Lq,Dv]
```

```python
q = torch.randn(2, 4, 5, 8)
k = torch.randn(2, 4, 7, 8)
v = torch.randn(2, 4, 7, 8)
output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
print(output.shape)  # [2, 4, 5, 8]
```

PyTorch 会按设备、dtype、shape 和 mask 选择 math、memory-efficient 或 Flash Attention 等后端；调用 SDPA 不保证必然使用某一个 kernel。

## 9. SDPA 参数、dropout 与 bool mask 语义

常用签名：

```python
F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
)
```

SDPA 不会读取外层 `.training` 状态，评估时必须显式传 `dropout_p=0.0`：

```python
def call_sdpa(q, k, v, training, dropout=0.1):
    return F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout if training else 0.0,
    )
```

SDPA 的 bool `attn_mask` 中 **True=允许，False=屏蔽**：

```python
q = torch.randn(1, 2, 4, 8)
k = torch.randn(1, 2, 5, 8)
v = torch.randn(1, 2, 5, 8)
allowed = torch.tensor([[True, True, True, False, False]])
mask = allowed[:, None, None, :]  # [B,1,1,Lk]
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
print(out.shape)
```

float mask 是加到 score 的 bias：

```python
bias = torch.zeros(1, 1, 4, 5)
bias[..., :, -1] = -2.0
out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
```

## 10. SDPA causal attention 与 GQA

自回归 causal attention：

```python
q = torch.randn(2, 4, 6, 8)
k = torch.randn(2, 4, 6, 8)
v = torch.randn(2, 4, 6, 8)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
print(out.shape)
```

通常不要同时传 `is_causal=True` 和自定义 `attn_mask`。非方形 Q/K（例如 KV cache）因果对齐应查当前版本文档并单测。

GQA 让 query head 多于 key/value head，可减少 KV cache：

```python
q = torch.randn(2, 8, 16, 64, device=device)
k = torch.randn(2, 2, 16, 64, device=device)
v = torch.randn(2, 2, 16, 64, device=device)

# 仅在当前版本、设备和后端支持时运行
out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, dropout_p=0.0)
print(out.shape)  # [2, 8, 16, 64]
```

## 11. 用 SDPA 封装完整 Self-Attention

```python
class SDPAMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, attn_mask=None, is_causal=False):
        # x: [B,L,D]；mask 遵循 SDPA: True=允许
        b, length, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(b, length, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, length, self.d_model)
        return self.out_proj(y)

model = SDPAMultiHeadSelfAttention(64, 4, dropout=0.1)
x = torch.randn(2, 8, 64)
print(model(x).shape)  # [2, 8, 64]
```

此示例只支持 self-attention。Cross-Attention 需要分别处理 query 长度 `Lq` 与 memory 长度 `Lk`。

## 12. SDPA 性能与 kernel 选择

SDPA 的融合实现可减少完整 attention matrix 的物化与显存读写，但标准全 attention 的数学复杂度仍约为 `O(L^2*D)`。

```python
if torch.cuda.is_available():
    q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    for _ in range(5):
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    torch.cuda.synchronize()
    print(y.shape)
```

性能检查应记录设备、dtype、序列长度、head dimension 与 mask 类型；先 warmup，再使用 profiler/benchmark 观察真实耗时。`torch.nn.attention.sdpa_kernel` 和 `SDPBackend` 的控制接口依版本变化，使用前应查当前文档。

## 13. `nn.TransformerEncoderLayer`：标准 Encoder 子层

Encoder layer 包含 self-attention、残差、LayerNorm 和 position-wise FFN：

```text
x -> self-attention -> residual + norm -> FFN -> residual + norm
```

```python
layer = nn.TransformerEncoderLayer(
    d_model=64,
    nhead=4,
    dim_feedforward=256,
    dropout=0.1,
    activation="gelu",
    batch_first=True,
    norm_first=True,
)
x = torch.randn(2, 10, 64)
y = layer(x)
print(y.shape)  # [2, 10, 64]
```

| 参数 | 含义 |
| --- | --- |
| `d_model` | token 表示维度 |
| `nhead` | head 数，`d_model` 必须整除 |
| `dim_feedforward` | FFN 中间宽度，常为 `4*d_model` |
| `activation` | ReLU/GELU 等 FFN 激活 |
| `batch_first` | 使用 `[B,L,E]` |
| `norm_first` | True 是 pre-norm；False 是 post-norm |

```python
print(inspect.signature(nn.TransformerEncoderLayer))
```

## 14. Encoder Layer 的 mask 与 padding

`src_mask` 约束 query-key 连边；`src_key_padding_mask` 标识 padding key。Transformer 模块中的 bool mask 通常是 **True=禁止/忽略**：

```python
L = 6
causal_block = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
x = torch.randn(2, L, 32)
layer = nn.TransformerEncoderLayer(32, 4, batch_first=True)
y = layer(x, src_mask=causal_block)
print(y.shape)
```

```python
token_ids = torch.tensor([[5, 7, 2, 0, 0], [4, 8, 1, 9, 2]])
x = torch.randn(2, 5, 32)
padding_mask = token_ids.eq(0)  # [B,S]，True=PAD
y = layer(x, src_key_padding_mask=padding_mask)
```

屏蔽 padding key 不代表 padding query 的输出自动为零；pooling、loss 和后续逻辑仍应忽略 padding 位置。

## 15. `nn.TransformerEncoder`：堆叠 Encoder Layer

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=256,
    batch_first=True, norm_first=True,
)
encoder = nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=6,
    norm=nn.LayerNorm(64),
    enable_nested_tensor=False,
)
x = torch.randn(2, 20, 64)
y = encoder(x)
print(y.shape)  # [2, 20, 64]
```

官方文档提示：堆叠层的参数初始值可能相同，虽然各层参数独立。构造后可以显式初始化：

```python
def init_linear_layers(module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)

init_linear_layers(encoder)
```

初始化只在训练前执行一次。`enable_nested_tensor` 属于性能优化开关，学习阶段可先关闭，实际收益应以当前版本和数据 padding 比例测试。

## 16. Encoder-only 文本分类模型

真实模型还需要 embedding、位置编码、pooling 和任务 head：

```python
class TinyEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, pad_id=0, d_model=64, nhead=4, max_len=256):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, token_ids):
        _, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        pad_mask = token_ids.eq(self.pad_id)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.head(pooled)

model = TinyEncoderClassifier(vocab_size=5000, num_classes=3)
ids = torch.tensor([[5, 18, 7, 2, 0], [4, 9, 10, 8, 3]])
print(model(ids).shape)  # [2, 3]
```

attention 本身没有顺序感，因此必须加入位置编码。全 padding 样本应在数据层避免。

## 17. `nn.TransformerDecoderLayer`：masked self-attention 加 cross-attention

标准 decoder layer 通常包含三段：

```text
tgt -> masked self-attention -> residual/norm
    -> cross-attention(memory) -> residual/norm
    -> FFN -> residual/norm
```

masked self-attention 防止目标位置读取未来 token；cross-attention 令 decoder query 从 encoder memory 的 key/value 中读取源序列信息。

```python
decoder_layer = nn.TransformerDecoderLayer(
    d_model=64,
    nhead=4,
    dim_feedforward=256,
    dropout=0.1,
    batch_first=True,
    norm_first=True,
)

tgt = torch.randn(2, 5, 64)     # [B,T,E]
memory = torch.randn(2, 8, 64)  # [B,S,E]，来自 encoder
out = decoder_layer(tgt=tgt, memory=memory)
print(out.shape)  # [2, 5, 64]
```

输出长度始终随 `tgt` 长度，不随 source memory 长度。

## 18. Decoder Layer 的四类 mask

`TransformerDecoderLayer.forward` 的常用 mask：

| 参数 | 作用 | 常见 shape |
| --- | --- | --- |
| `tgt_mask` | target self-attention 的连接约束 | `[T,T]` |
| `memory_mask` | target query 到 source memory 的连接约束 | `[T,S]` |
| `tgt_key_padding_mask` | target 中的 PAD key | `[B,T]` |
| `memory_key_padding_mask` | encoder memory 中的 PAD key | `[B,S]` |

```python
B, T, S, E = 2, 5, 8, 64
tgt = torch.randn(B, T, E)
memory = torch.randn(B, S, E)
tgt_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
tgt_padding = torch.tensor([
    [False, False, False, False, True],
    [False, False, False, False, False],
])
memory_padding = torch.tensor([
    [False, False, False, False, False, True, True, True],
    [False, False, False, False, False, False, False, False],
])

decoder_layer = nn.TransformerDecoderLayer(E, 4, batch_first=True)
out = decoder_layer(
    tgt=tgt,
    memory=memory,
    tgt_mask=tgt_mask,
    tgt_key_padding_mask=tgt_padding,
    memory_key_padding_mask=memory_padding,
)
print(out.shape)
```

Transformer 的 bool mask 语义一般是 `True=禁止`，不要照搬 SDPA 的 `True=允许`。

## 19. `nn.TransformerDecoder`：堆叠 Decoder Layer

```python
decoder_layer = nn.TransformerDecoderLayer(
    d_model=64, nhead=4, dim_feedforward=256,
    batch_first=True, norm_first=True,
)
decoder = nn.TransformerDecoder(
    decoder_layer=decoder_layer,
    num_layers=4,
    norm=nn.LayerNorm(64),
)

tgt = torch.randn(2, 6, 64)
memory = torch.randn(2, 9, 64)
tgt_mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
out = decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
print(out.shape)  # [2, 6, 64]
```

`TransformerDecoder` 不包含 token embedding、位置编码、词表投影、beam search 或 KV cache；它只是 decoder layer 堆叠。

## 20. `nn.Transformer`：完整 Encoder-Decoder 骨架

`nn.Transformer` 内部包含标准 encoder 和 decoder 堆叠，但仍然不处理 token id、embedding、位置编码、输出 head 或训练目标：

```python
transformer = nn.Transformer(
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    batch_first=True,
    norm_first=True,
)

src = torch.randn(2, 7, 64)
tgt = torch.randn(2, 5, 64)
tgt_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
out = transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
print(out.shape)  # [2, 5, 64]
```

`src` 和 `tgt` 必须已经是 embedding 加位置编码后的连续表示。

## 21. Teacher Forcing 与 target shift-right

训练自回归 decoder 时，不能把当前答案 token 直接作为该位置输入。应右移 target：

```text
完整目标:     <BOS> 我 喜欢 PyTorch <EOS>
decoder 输入: <BOS> 我 喜欢 PyTorch
训练标签:      我 喜欢 PyTorch <EOS>
```

```python
bos_id, eos_id, pad_id = 1, 2, 0
full_target = torch.tensor([[bos_id, 15, 28, 41, eos_id, pad_id]])
decoder_input = full_target[:, :-1]
target_labels = full_target[:, 1:]
print(decoder_input)
print(target_labels)

T = decoder_input.size(1)
tgt_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
```

即使输入已右移，也需要 causal mask，防止早期位置读取 decoder 输入中的未来词。

## 22. 一个基于 `nn.Transformer` 的 Seq2Seq 模型

```python
class TinySeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, max_len=128, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=2, num_decoder_layers=2,
            batch_first=True,
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_input_ids):
        _, src_len = src_ids.shape
        _, tgt_len = tgt_input_ids.shape
        src_pos = torch.arange(src_len, device=src_ids.device).unsqueeze(0)
        tgt_pos = torch.arange(tgt_len, device=tgt_input_ids.device).unsqueeze(0)
        src = self.src_emb(src_ids) + self.pos_emb(src_pos)
        tgt = self.tgt_emb(tgt_input_ids) + self.pos_emb(tgt_pos)
        tgt_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
            diagonal=1,
        )
        h = self.transformer(
            src=src, tgt=tgt, tgt_mask=tgt_mask,
            src_key_padding_mask=src_ids.eq(self.pad_id),
            tgt_key_padding_mask=tgt_input_ids.eq(self.pad_id),
            memory_key_padding_mask=src_ids.eq(self.pad_id),
        )
        return self.lm_head(h)  # [B,T,V]

model = TinySeq2Seq(vocab_size=100)
src_ids = torch.tensor([[5, 8, 2, 0]])
full_target = torch.tensor([[1, 9, 17, 2, 0]])
logits = model(src_ids, full_target[:, :-1])
loss = F.cross_entropy(logits.transpose(1, 2), full_target[:, 1:], ignore_index=0)
print(logits.shape, loss.item())
```

这只是训练前向示意，不包含生成时逐 token 解码、beam search、KV cache 或词表构建。

## 23. FlexAttention：可编程 attention 内核

FlexAttention 是较新的 PyTorch API，面向需要特殊 score、mask 或结构化稀疏 attention 的进阶场景。它通常接收已拆 head 的 Q/K/V：

```text
Q/K/V: [B,H,L,Dh]
```

它允许定义：

- `score_mod`：如何修改每条 query-key 边的 score，例如相对位置 bias、soft-capping。
- `mask_mod`：某条 query-key 边是否存在，例如 causal、滑动窗口、文档边界。
- `BlockMask`：将稀疏规则变成块结构，让内核跳过不必要 block。

基础调用通常需要 CUDA 和 `torch.compile`，具体支持依版本和硬件而异：

```python
from torch.nn.attention.flex_attention import flex_attention

if torch.cuda.is_available():
    q = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    compiled_flex_attention = torch.compile(flex_attention)
    out = compiled_flex_attention(q, k, v)
    print(out.shape)  # [2, 4, 128, 64]
```

FlexAttention 不是 `nn.Module`，不会自动完成投影、残差、Norm 或 FFN。对于普通 padding/causal attention，SDPA 与 Transformer 内置层通常更直接。

## 24. FlexAttention 的 `score_mod`

`score_mod(score, batch, head, q_idx, kv_idx)` 返回修改后的 score。相对位置距离 bias：

```python
from torch.nn.attention.flex_attention import flex_attention

def relative_position_bias(score, batch, head, q_idx, kv_idx):
    distance = (q_idx - kv_idx).abs()
    return score - 0.01 * distance.to(score.dtype)

if torch.cuda.is_available():
    q = torch.randn(1, 2, 32, 16, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.compile(flex_attention)(q, k, v, score_mod=relative_position_bias)
    print(out.shape)
```

soft-capping 示例：

```python
def softcap_score(score, batch, head, q_idx, kv_idx):
    cap = 20.0
    return cap * torch.tanh(score / cap)
```

`score_mod` 必须满足当前编译器对可追踪函数的限制；避免数据依赖 Python 分支、CPU 拷贝和不支持操作。

## 25. FlexAttention 的 `mask_mod` 和 BlockMask

`mask_mod(batch, head, q_idx, kv_idx)` 返回 bool，其中 True 表示允许 attention。构造 causal BlockMask：

```python
from torch.nn.attention.flex_attention import create_block_mask

def causal_mask_mod(batch, head, q_idx, kv_idx):
    return q_idx >= kv_idx

if torch.cuda.is_available():
    B, H, L, Dh = 2, 4, 256, 32
    block_mask = create_block_mask(
        causal_mask_mod,
        B=B, H=H, Q_LEN=L, KV_LEN=L,
        device="cuda",
    )
    q = torch.randn(B, H, L, Dh, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
    print(out.shape)
```

滑动窗口 causal rule：

```python
window = 64

def causal_window_mask(batch, head, q_idx, kv_idx):
    return (kv_idx <= q_idx) & (kv_idx >= q_idx - window + 1)
```

规则稀疏不代表一定加速：序列过短、BlockMask 太稠密或模式难以按块跳过时，编译/调度成本可能大于收益。

## 26. FlexAttention 的文档隔离示例

packed documents 中可限制 token 只能查看同一文档：

```python
document_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]], device="cuda")

def document_causal_mask(batch, head, q_idx, kv_idx):
    same_doc = document_ids[batch, q_idx] == document_ids[batch, kv_idx]
    return same_doc & (kv_idx <= q_idx)
```

生产中要验证该闭包捕获的 Tensor 能否被 `torch.compile` 追踪，并确保 device、一致 batch 维与动态 shape 规则正确。简单任务优先选择内置 padding/causal mask。

## 27. FlexAttention API 演进与排错

FlexAttention 仍在快速演进。先检查当前安装版本而不是复制旧代码：

```python
import inspect
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

print(inspect.signature(flex_attention))
print(inspect.signature(create_block_mask))
```

常见问题：

- `ImportError`：版本不含 FlexAttention，或导入路径不同。
- 编译/backend error：CUDA、硬件、dtype 或 score/mask 函数不受支持。
- 首轮很慢：`torch.compile` 有编译开销，性能测试前应 warmup。
- 与 SDPA 不同：检查 scale、dropout、mask 规则和 score_mod。
- 没有加速：BlockMask 没跳过足够 block，或序列过短。

## 28. SDPA、MHA、Transformer Layer、FlexAttention 对比

| 能力 | MHA | SDPA | Transformer Layer | FlexAttention |
| --- | --- | --- | --- | --- |
| 自动 Q/K/V 投影 | 是 | 否 | 是 | 否 |
| 自动 split/merge head | 是 | 否 | 是 | 否 |
| residual/FFN/Norm | 否 | 否 | 是 | 否 |
| 返回 attention 权重 | 是，可选 | 否 | 否 | 通常不是目标 |
| 自定义 score | 有限 | 有限 | 有限 | 强 |
| 稀疏 block 支持 | 有限 | 有限 | 有限 | 强 |
| 推荐阶段 | 基础/通用 | 进阶模块 | 标准模型 | 高级性能/研究 |

## 29. mask 语义速查表

| API | bool mask 的 True 通常表示 |
| --- | --- |
| `F.scaled_dot_product_attention(attn_mask=...)` | 允许 attention |
| `nn.MultiheadAttention(attn_mask=...)` | 禁止 attention |
| `nn.MultiheadAttention(key_padding_mask=...)` | 忽略 key/padding |
| `nn.Transformer*` 的 `*_mask` | 禁止 attention |
| `nn.Transformer*` 的 `*_key_padding_mask` | 忽略 key/padding |
| FlexAttention `mask_mod` 返回值 | 允许 attention |

float mask 通常作为 score bias 加入；绝对屏蔽常使用 `-inf`。每一行至少需要一个合法 key，否则 softmax 可能产生 NaN。

## 30. 训练、评估与调试模板

Transformer 模块不是完整任务模型，还需 head、标签、loss 与 optimizer：

```python
model = TinyEncoderClassifier(vocab_size=2000, num_classes=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

token_ids = torch.randint(1, 2000, (16, 32), device=device)
labels = torch.randint(0, 3, (16,), device=device)

model.train()
logits = model(token_ids)
loss = criterion(logits, labels)
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
print(loss.item())

model.eval()
with torch.inference_mode():
    predictions = model(token_ids).argmax(dim=-1)
```

最小 shape/mask 自测：

```python
def smoke_test_mha():
    mha = nn.MultiheadAttention(24, 3, batch_first=True)
    x = torch.randn(2, 5, 24)
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    y, w = mha(x, x, x, attn_mask=mask, average_attn_weights=False)
    assert y.shape == x.shape
    assert w.shape == (2, 3, 5, 5)
    assert w[..., 0, 1:].abs().max() < 1e-6
    print("MHA smoke test passed")

smoke_test_mha()
```

## 31. 常见错误与练习路线

| 现象 | 常见原因 | 修复 |
| --- | --- | --- |
| 输出 shape 异常 | `batch_first` 混用 | 固定项目布局并打印 shape |
| 生成任务验证异常好 | causal mask 缺失或 target 未右移 | 检查 `triu(..., diagonal=1)` 与 shift-right |
| padding 污染输出 | 漏传 padding mask | Transformer/MHA 用 `True=忽略` |
| SDPA/MHA 不一致 | bool mask 语义相反，dropout 不同 | 统一 mask，先设 dropout=0 |
| NaN | 某 query 全被屏蔽 | 检查每行至少一个合法 key |
| FlexAttention 无加速 | 编译开销、序列太短或 block 太稠密 | warmup 后 benchmark，并检查稀疏度 |

建议实践顺序：

1. 使用 MHA 做 self-attention，并查看 per-head 权重。
2. 使用 MHA 做 cross-attention，验证输出长度跟 query 对齐。
3. 分别实现 padding/causal mask，验证被禁止权重为零。
4. 使用 SDPA 手写 projection 与 split/merge heads。
5. 用 EncoderLayer/Encoder 做文本分类骨架。
6. 用 DecoderLayer 与 `nn.Transformer` 做 teacher forcing 前向。
7. 在 CUDA 环境用 FlexAttention 实现 causal 与滑动窗口 block mask。
8. 最后才对比长序列性能、混合精度与不同 kernel。

## 32. 总结

- `nn.MultiheadAttention` 适合标准 attention 层和权重观察。
- `F.scaled_dot_product_attention` 适合自定义模块中的高效核心计算。
- `TransformerEncoderLayer` / `TransformerDecoderLayer` 提供标准 Attention + FFN + residual + norm。
- `TransformerEncoder` / `TransformerDecoder` 负责堆叠，不负责 embedding、位置编码、任务 head 和生成策略。
- FlexAttention 面向自定义 score/mask/稀疏 block，需要实际环境验证编译与性能。

最可靠的工作流是：明确 shape，先用小 batch 验证 mask 语义与输出，再训练，最后才做长序列和性能优化。
