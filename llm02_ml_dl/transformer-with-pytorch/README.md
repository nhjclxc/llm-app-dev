# Transformer with PyTorch：项目分析

## 1. 项目定位

这是一个**从零手写 Encoder-Decoder Transformer** 的英译意机器翻译教学项目。它没有使用 `nn.Transformer` 或 `nn.MultiheadAttention`，而是在 [model.py](model.py) 中实现了词嵌入、正弦位置编码、LayerNorm、多头注意力、前馈网络、残差连接、Encoder、Decoder 和词表投影层。

训练数据来自 Hugging Face `opus_books` 数据集的 `en-it` 配置；源语言是英语（`en`），目标语言是意大利语（`it`）。项目根目录已包含训练出的 BPE tokenizer 文件：

```text
transformer-with-pytorch/
├── config.py              # 超参数、数据集与权重路径
├── dataset.py             # 双语数据集、padding、source/target mask
├── model.py               # 手写 Transformer 模型
├── train.py               # tokenizer、数据加载、训练、checkpoint 保存
├── tokenizer_en.json      # 英语 BPE tokenizer
└── tokenizer_it.json      # 意大利语 BPE tokenizer
```

当前代码只实现了训练，不包含独立的验证指标、贪婪/beam 解码、BLEU 等翻译评估或推理命令。

## 2. 默认配置

`config.py` 中的默认配置如下：

| 配置 | 值 | 作用 |
| --- | ---: | --- |
| `datasource` | `opus_books` | Hugging Face 数据集名 |
| `lang_src` / `lang_tgt` | `en` / `it` | 英语到意大利语翻译 |
| `batch_size` | 8 | 训练 batch 大小 |
| `num_epochs` | 20 | 训练轮数 |
| `lr` | `1e-4` | Adam 学习率 |
| `seq_len` | 350 | 输入/输出固定最大长度 |
| `d_model` | 512 | token 表示维度 |
| Transformer layers | 6 | `build_transformer` 默认参数 `N=6` |
| attention heads | 8 | 每 head 维度为 64 |
| FFN hidden dim | 2048 | 前馈网络中间层维度 |
| dropout | 0.1 | embedding、attention、FFN/残差位置使用 |

模型配置基本对应论文 *Attention Is All You Need* 的 base 级别结构，但训练 batch 很小、没有学习率 warmup/schedule，且数据规模和硬件限制会显著影响效果。

## 3. 从原始句子到训练 batch 的数据流

训练数据流如下：

```text
OPUS Books 英/意平行句对
        ↓
为英语和意大利语分别训练/加载 BPE tokenizer
        ↓
BilingualDataset 编码、加入 [SOS]/[EOS]、padding 到 seq_len=350
        ↓
encoder_input / decoder_input / label / encoder_mask / decoder_mask
        ↓
Transformer encoder-decoder
        ↓
[B, seq_len, target_vocab_size] logits
        ↓
CrossEntropyLoss（忽略 target PAD）
```

### 3.1 Tokenizer

`train.py` 的 `get_or_build_tokenizer`：

1. 使用每种语言全部训练语料构建独立 BPE tokenizer；
2. 使用 `Whitespace()` 预分词；
3. 设定 `[UNK]`、`[PAD]`、`[SOS]`、`[EOS]` 四个特殊 token；
4. `min_frequency=2`；
5. 如果 `tokenizer_en.json` / `tokenizer_it.json` 已存在，直接加载而不是重新训练。

训练 tokenizer 的代码：

```python
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
    min_frequency=2,
)
tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
```

Tokenizer 在切分 train/validation 前使用了所有句子，因此验证句子的文本参与了**词表学习**。这通常不会泄漏翻译标签或造成训练数据直接记忆，但严格实验中应只在训练子集上训练 tokenizer，以避免验证文本影响 vocabulary/BPE merges。

### 3.2 单个样本格式

对一个源句 `src` 和目标句 `tgt`，`BilingualDataset.__getitem__` 生成：

```text
encoder_input = [SOS] src_tokens [EOS] [PAD] ...
decoder_input = [SOS] tgt_tokens       [PAD] ...
label         =       tgt_tokens [EOS] [PAD] ...
```

这就是 teacher forcing 的右移形式：decoder 在第 `t` 步读取前缀，预测标签中同位置的下一个目标 token。

若任一编码后的句子超过 `seq_len`，数据集会直接抛出 `ValueError`，不会截断：

```python
if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
    raise ValueError("句子太长, 超过了seq_len, 输入不全会翻译的不准")
```

### 3.3 Mask

返回张量及预期 shape：

| 字段 | shape | 用途 |
| --- | --- | --- |
| `encoder_input` | `[L]` | 源语言 token id |
| `decoder_input` | `[L]` | 右移后的目标 token id |
| `label` | `[L]` | 当前步预测目标 |
| `encoder_mask` | `[1, 1, L]` | 屏蔽 source PAD key |
| `decoder_mask` | `[1, L, L]` | 同时屏蔽 target PAD 与未来 token |

经 `DataLoader` 拼 batch 后：

```text
encoder_input: [B, L]
encoder_mask:  [B, 1, 1, L]
decoder_mask:  [B, 1, L, L]
```

模型内部的 attention 使用约定：`1 / True` 表示允许注意，`0 / False` 表示 `masked_fill(..., -1e9)` 屏蔽。Encoder mask 可广播到 `[B, H, L, L]`；Decoder mask 结合 padding mask 和下三角 causal mask，避免模型偷看未来目标词。

## 4. 模型结构分析

`build_transformer` 构建的网络为：

```text
source token ids
  -> InputEmbeddings * sqrt(d_model)
  -> sinusoidal positional encoding
  -> 6 x EncoderBlock
  -> encoder memory

target token ids（右移）
  -> InputEmbeddings * sqrt(d_model)
  -> sinusoidal positional encoding
  -> 6 x DecoderBlock
       - masked self-attention
       - encoder-decoder cross-attention
       - feed-forward network
  -> Linear(d_model, target_vocab_size)
  -> target token logits
```

### 4.1 `InputEmbeddings`

`nn.Embedding(vocab_size, d_model)` 将 token id 映射为向量，并乘以 `sqrt(d_model)`：

```python
return self.embedding(x) * math.sqrt(self.d_model)
```

这是原始 Transformer 论文中的缩放方式，用来让 token embedding 与位置编码的数值量级更协调。

### 4.2 `PositionalEncoding`

模型使用固定正弦/余弦位置编码，预先创建 `[1, seq_len, d_model]` 的非训练 buffer `pe`。它会随 checkpoint 保存，但不参与优化：

```python
self.register_buffer("pe", pe)
x = x + self.pe[:, :x.shape[1], :]
```

这一实现正确表达了 token 顺序；纯 attention 不带位置编码时无法识别顺序。

### 4.3 `MultiHeadAttentionBlock`

手写 MHA 做了以下工作：

1. 以独立 `w_q`、`w_k`、`w_v` 线性层投影 Q/K/V；
2. 将 `[B,L,d_model]` reshape 为 `[B,H,L,d_k]`；
3. 计算 `QK^T / sqrt(d_k)`；
4. 在 softmax 前应用 mask；
5. 对 attention score 做 dropout；
6. 对 V 加权求和、合并 heads、通过输出投影 `w_o`。

核心公式对应：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Encoder 中是 self-attention：`(x, x, x, src_mask)`；Decoder 中首先是 masked self-attention，随后是 cross-attention：

```python
self_attention_block(x, x, x, tgt_mask)
cross_attention_block(x, encoder_output, encoder_output, src_mask)
```

### 4.4 FFN、Residual 与 LayerNorm

每层 FFN 的结构是：

```text
Linear(512, 2048) -> ReLU -> Dropout -> Linear(2048, 512)
```

`ResidualConnection` 实际实现为 pre-norm：

```python
return x + self.dropout(sublayer(self.norm(x)))
```

这和原始论文常见的 post-norm 图不同，但属于合理、常用的现代稳定训练方式。Encoder 和 Decoder 末尾都再进行一次 LayerNorm。

## 5. 训练过程分析

`train_model` 的主要步骤：

1. 下载/加载平行数据；
2. 创建或加载 BPE tokenizer；
3. 使用 `random_split` 按 90% / 10% 划分训练和验证数据；
4. 生成 `BilingualDataset` 与 DataLoader；
5. 创建 Transformer；
6. 使用 Adam 与带 label smoothing 的交叉熵训练；
7. 每个 epoch 保存 checkpoint 与 TensorBoard training loss。

损失函数为：

```python
loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer_src.token_to_id("[PAD]"),
    label_smoothing=0.1,
)
```

输出 logits 的 shape 是 `[B,L,V_tgt]`，训练前展平为 `[B*L,V_tgt]`；label 展平为 `[B*L]`：

```python
loss = loss_fn(
    proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
    label.view(-1),
)
```

这种 logits 与 `CrossEntropyLoss` 的组合是正确的：ProjectionLayer 未提前做 softmax，交叉熵内部会处理数值稳定的 log-softmax。

Checkpoint 内容：

```text
epoch
model_state_dict
optimizer_state_dict
global_step
```

保存位置为：

```text
opus_books_weights/tmodel_XX.pt
```

TensorBoard 日志写入 `runs/tmodel`，可在仓库根目录运行：

```bash
tensorboard --logdir runs
```

## 6. 当前代码的关键问题与改进建议

### 6.1 验证集被创建但完全没有使用

严重程度：高。

`get_ds` 返回 `val_dataloader`，但训练函数从不迭代它。因此项目当前没有：

- validation loss；
- 最佳 checkpoint 选择；
- early stopping；
- 翻译质量指标（BLEU/chrF）；
- 基于验证集的超参数选择依据。

现在每个 epoch 都保存最后状态，而非最佳状态。至少应增加评估函数，并按 validation loss 保存最佳模型：

```python
@torch.inference_mode()
def evaluate_loss(model, dataloader, loss_fn, device, vocab_size):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        enc = batch["encoder_input"].to(device)
        dec = batch["decoder_input"].to(device)
        label = batch["label"].to(device)
        enc_mask = batch["encoder_mask"].to(device)
        dec_mask = batch["decoder_mask"].to(device)

        logits = model.project(model.decode(model.encode(enc, enc_mask), enc_mask, dec, dec_mask))
        loss = loss_fn(logits.reshape(-1, vocab_size), label.reshape(-1))
        non_pad = label.ne(loss_fn.ignore_index).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    return total_loss / max(total_tokens, 1)
```

### 6.2 目标语言特殊 token 错误地从 source tokenizer 创建

严重程度：高，但当前 tokenizer 文件可能因特殊 token 注册顺序一致而侥幸运行。

`BilingualDataset.__init__` 中的 `[SOS]`、`[EOS]`、`[PAD]` 都通过 `tokenizer_src` 获取：

```python
self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")])
self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")])
self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")])
```

但 decoder input 与 label 是目标语言 token id，必须使用 `tokenizer_tgt` 的特殊 token id；loss 的 `ignore_index` 也应使用 target tokenizer。正确设计应分开 source/target token：

```python
self.src_sos = torch.tensor([tokenizer_src.token_to_id("[SOS]")])
self.src_eos = torch.tensor([tokenizer_src.token_to_id("[EOS]")])
self.src_pad = torch.tensor([tokenizer_src.token_to_id("[PAD]")])
self.tgt_sos = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")])
self.tgt_eos = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")])
self.tgt_pad = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")])
```

在当前 BPE trainer 中，两种 tokenizer 都以相同顺序声明 special tokens，因此它们的 id **大概率**相同；但这只是隐含不变量，不应让模型正确性依赖它。

对应地，损失应修改为：

```python
loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
    label_smoothing=0.1,
)
```

### 6.3 `random_split` 未设置随机种子

严重程度：中。

每次运行会得到不同 train/validation 切分，难以复现实验和比较 checkpoint。建议：

```python
generator = torch.Generator().manual_seed(42)
train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size], generator=generator)
```

还应统一设置 Python、PyTorch、CUDA 的随机种子，并记录配置与 tokenizer 版本。

### 6.4 checkpoint 只保存，不支持恢复训练

严重程度：中。

代码保存了 model、optimizer、epoch 和 global step，却没有 `preload` 配置或加载分支。重新运行只能从头训练。恢复训练时应加载 checkpoint，并从 `epoch + 1` 继续：

```python
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
initial_epoch = checkpoint["epoch"] + 1
global_step = checkpoint["global_step"]
```

### 6.5 GPU 信息判断错误

严重程度：低。

`device` 是 `torch.device` 对象，但代码比较的是字符串：

```python
if device == "cuda":
```

这通常为 False，因此 GPU 名称和显存不会打印。可写为：

```python
if device.type == "cuda":
    print(torch.cuda.get_device_name(device))
```

### 6.6 LayerNorm 方差实现与标准 `nn.LayerNorm` 有轻微差异

严重程度：低到中。

自定义 `LayerNormalization` 使用：

```python
std = x.std(dim=-1, keepdim=True)
```

`torch.std` 默认使用样本标准差（历史版本/默认行为需确认），而标准 LayerNorm 通常使用总体方差形式。更接近标准实现且避免额外开方的写法是：

```python
var = x.var(dim=-1, keepdim=True, unbiased=False)
return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias
```

这不会让项目无法训练，但会造成与 PyTorch `nn.LayerNorm` 和论文实现的数值差异。

### 6.7 长句策略、数据效率和训练稳定性不足

严重程度：中，取决于数据集。

- 长度超过 350 token 的样本会令训练直接失败；生产训练通常过滤、截断或采用动态 batch。
- 全部样本 padding 到 350，短句会造成明显计算浪费；可按长度 bucket 或使用动态 padding。
- 没有 learning-rate warmup/scheduler；原始 Transformer 对优化器与 warmup 较敏感。
- 没有 gradient clipping，遇到不稳定梯度时没有保护。
- `val_dataloader` 设置 `shuffle=True`，即使将来做 validation loss 也不必要；应为 False。
- 未设置 `num_workers` / `pin_memory`，数据加载性能未优化。

## 7. 如何运行

需要的核心依赖：

```bash
pip install torch datasets tokenizers tensorboard tqdm
```

从项目目录运行，确保 `from model import ...` 等本地导入可解析：

```bash
cd transformer-with-pytorch
python train.py
```

首次运行会下载 `opus_books` 数据集；若 tokenizer 文件不存在，还会构建 BPE tokenizer。训练过程会下载数据、占用本地缓存，并在当前工作目录创建：

```text
opus_books_weights/
runs/tmodel/
```

如果从仓库根目录用模块方式执行，应先将导入改为包内相对导入，或将目录整理成 Python package；当前代码更适合直接在本目录运行。

## 8. 建议的演进顺序

1. 修复 source/target special token 与 target PAD loss 的归属；
2. 增加验证 loss 与最佳 checkpoint；
3. 增加贪婪解码和若干翻译样例；
4. 使用 BLEU/chrF 等指标评估验证集；
5. 支持 checkpoint 恢复、固定随机种子和实验配置记录；
6. 按长度分桶或动态 padding，减少 `seq_len=350` 的无效计算；
7. 增加 warmup、scheduler、gradient clipping；
8. 将手写 `MultiHeadAttentionBlock` 与 `F.scaled_dot_product_attention` 或 `nn.MultiheadAttention` 对照，验证数值和性能。

## 9. 结论

该项目的教育价值较高：它把 Transformer 的关键结构完整拆开，数据集构造也正确体现了 encoder 输入、decoder 右移输入、label、padding mask 和 causal mask 的关系。Encoder self-attention、Decoder masked self-attention、cross-attention、FFN、pre-norm residual 和词表投影的主流程是连贯的。

但它目前更适合作为“理解 Transformer 翻译训练流程”的示例，而非可直接用于严格实验或可复现训练的完整项目。最需要优先处理的是：**目标 tokenizer special token 的正确归属、validation 实际使用、最佳模型选择和 checkpoint 恢复**。完成这些修正后，再补充解码和 BLEU 等评估，才形成完整的机器翻译训练/验证/推理闭环。
