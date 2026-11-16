# Embedding 嵌入：从 `nn.Embedding` 到语义向量与向量数据库

Embedding（嵌入）是把离散对象映射到连续向量空间的表示方法。词、子词、用户 ID、商品 ID、类别编号、图节点都可以映射为稠密向量，让神经网络通过向量运算学习对象之间的关系。

本文按从基础到进阶的顺序学习：离散 ID 与 one-hot、Embedding 查表原理、`nn.Embedding`、梯度与稀疏更新、文本/推荐/类别 embedding、预训练向量、向量相似度与近邻检索、FAISS/向量数据库，以及面向 RAG 的检索流程。先要记住一个重要区分：

```text
nn.Embedding：模型中的可训练参数表，输入整数 ID，查出稠密向量。
向量数据库：保存文档/对象向量并建立近邻索引，用于相似度检索和元数据过滤。
```

二者都处理向量，但不是同一种组件：Embedding 层负责“表示如何学习/查询”，向量数据库负责“如何存储和检索大量已经生成的向量”。

示例默认适用于 PyTorch 2.x；向量数据库部分会区分纯 NumPy 原理示例、FAISS 与 Chroma 等可选依赖。

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__, device)
```

## 1. 学习路线与先修知识

| 阶段 | 核心问题 | 内容 |
| --- | --- | --- |
| 入门 | 离散 ID 为什么不能直接表示语义？ | 2-4 |
| 基础 | Embedding 表是什么，如何查表？ | 5-9 |
| 训练 | 梯度怎样更新 embedding，如何处理 PAD？ | 10-14 |
| 应用 | 文本、类别、用户/商品 embedding 怎么用？ | 15-20 |
| 进阶表示 | 预训练、共享、初始化和表示诊断 | 21-25 |
| 检索 | 相似度、近邻搜索、索引和向量数据库 | 26-34 |
| 系统实践 | RAG 检索数据流、更新、过滤和评估 | 35-40 |

建议先会 Tensor 索引、矩阵乘法、`nn.Module`、loss、反向传播和 `DataLoader`。Embedding 输入是整数索引，不是浮点特征；它的权重虽是浮点数，但查表操作本身不对 ID 求梯度。

## 2. 什么是离散数据

离散数据的取值来自有限或可枚举集合。例如：

```text
词：        cat, dog, run
用户：      user_17, user_29
商品类别：  book, clothing, food
类别标签：  0, 1, 2
```

数字 ID 只是对象的编号。编号之间的大小或距离通常没有含义：类别 `2` 不一定比类别 `1` “更大”，词 ID 100 和 101 也不一定语义接近。

```python
token_to_id = {"cat": 0, "dog": 1, "run": 2}
ids = torch.tensor([token_to_id["cat"], token_to_id["run"]])
print(ids)  # tensor([0, 2])，只是查表索引
```

把 ID 当连续数值直接喂给模型会暗中引入错误的顺序假设：

```python
ids_as_float = torch.tensor([[0.0], [1.0], [2.0]])
linear = nn.Linear(1, 4)
print(linear(ids_as_float))  # 数值相邻被当作有线性关系
```

## 3. One-hot 表示及其局限

词表大小为 `V` 时，ID `i` 的 one-hot 向量是长度 `V` 的向量，只有第 `i` 位为 1：

```python
ids = torch.tensor([0, 2, 1])
one_hot = F.one_hot(ids, num_classes=4).float()
print(one_hot)
print(one_hot.shape)  # [N=3, V=4]
```

one-hot 的优点是无歧义，每个 ID 有唯一表示；缺点是高维、稀疏，并且任意两个不同 ID 的向量距离都相同，不能表达“猫”和“狗”比“猫”和“微积分”更相似。

若后面接一个线性层 `W`：

$$
\operatorname{onehot}(i)W=W_{i,:}
$$

矩阵乘法结果等价于直接取 `W` 的第 `i` 行，因此 Embedding 查表就是高效的 one-hot 线性变换。

```python
V, D = 4, 3
W = torch.arange(V * D, dtype=torch.float32).reshape(V, D)
ids = torch.tensor([1, 3])
via_one_hot = F.one_hot(ids, num_classes=V).float() @ W
via_lookup = W[ids]
print(torch.equal(via_one_hot, via_lookup))  # True
```

## 4. Embedding 的核心思想

Embedding 为每个离散 ID 学习一个 `D` 维稠密向量：

$$
E\in\mathbb{R}^{V\times D},\qquad \operatorname{embedding}(i)=E_i
$$

- `V`：词表/实体总数。
- `D`：embedding dimension，每个对象的向量维度。
- `E`：embedding table，整个查找矩阵。
- `E_i`：ID `i` 对应的向量。

例如 5 个 token、每个用 3 个浮点数表示，Embedding 表就是 `[5,3]`：

```python
embedding_table = nn.Embedding(num_embeddings=5, embedding_dim=3)
token_ids = torch.tensor([0, 3, 4])
vectors = embedding_table(token_ids)
print(embedding_table.weight.shape)  # [5, 3]
print(vectors.shape)                 # [3, 3]
```

Embedding 向量中的数值通常不是人工指定的语义特征，而是通过训练目标学习得到。只有当训练目标或预训练过程让相似对象产生相近表示时，向量空间才会呈现有用的语义结构。

## 5. Embedding 是查表，不是神秘的神经网络

对整数 ID `i`，`nn.Embedding` 直接返回参数矩阵的第 `i` 行：

```python
table = nn.Embedding(6, 2)
ids = torch.tensor([4, 1, 4])
vectors = table(ids)

print(torch.allclose(vectors[0], vectors[2]))  # 同一 ID 查到同一行
print(torch.allclose(vectors[0], table.weight[4]))  # 与权重表第 4 行相同
```

查表不含对 ID 的连续插值，也不会因为 ID 从 4 变为 5 就保证向量只改变一点。ID 需要是合法的整数索引：

```python
ids = torch.tensor([[1, 2, 0], [3, 4, 1]], dtype=torch.long)
embedding = nn.Embedding(num_embeddings=5, embedding_dim=8)
out = embedding(ids)
print(out.shape)  # [B=2, L=3, D=8]
```

输入 dtype 通常用 `torch.long` / `torch.int64`；ID 最小为 0，最大必须小于 `num_embeddings`。

## 6. `nn.Embedding` 最小例子

构造参数最重要的两个是 `num_embeddings` 和 `embedding_dim`：

```python
vocab_size = 100
embedding_dim = 16
token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
)

token_ids = torch.tensor([7, 21, 7, 99])
token_vectors = token_embedding(token_ids)
print(token_vectors.shape)  # [4, 16]
```

`num_embeddings` 是可能的 ID 数量，不是 batch size，也不是序列长度。`embedding_dim` 是每个 ID 输出的特征维数。

## 7. 输入 shape 与输出 shape

`nn.Embedding` 接收任意 shape 的整型 ID Tensor，并在末尾追加 embedding dimension：

```text
input:  [*]        # 任意数量的 ID 维度
output: [*, D]     # 原 shape 后追加 embedding_dim
```

常见输入输出：

```python
embedding = nn.Embedding(1000, 32)

scalar_id = torch.tensor(8)
sentence_ids = torch.tensor([4, 9, 2, 0])
batch_ids = torch.randint(0, 1000, (3, 5))

print(embedding(scalar_id).shape)     # [32]
print(embedding(sentence_ids).shape)  # [4, 32]
print(embedding(batch_ids).shape)     # [3, 5, 32]
```

嵌入序列常记作 `[B,L,D]`，可直接送入 RNN、CNN、Transformer 的后续层。

## 8. `nn.Embedding` 权重是什么

Embedding 的权重是形状 `[V,D]` 的可训练参数：

```python
embedding = nn.Embedding(10, 4)
print(embedding.weight.shape)
print(embedding.weight.requires_grad)  # True
print(sum(p.numel() for p in embedding.parameters()))  # 40
```

默认权重通常从随机分布初始化。训练时，loss 对被使用的行产生梯度，optimizer 更新这些向量。

```python
ids = torch.tensor([2, 4, 2])
vectors = embedding(ids)
loss = vectors.square().mean()
loss.backward()
print(embedding.weight.grad.shape)  # [10, 4]
```

## 9. Embedding 与 Linear 层的等价关系

当 one-hot 作为 Linear 输入且 Linear 无 bias 时，输出等于对应 embedding 行。`nn.Embedding` 避免构造大型 one-hot Tensor：

```python
vocab_size, dim = 7, 5
linear = nn.Linear(vocab_size, dim, bias=False)
embedding = nn.Embedding(vocab_size, dim)

with torch.no_grad():
    embedding.weight.copy_(linear.weight.T)

ids = torch.tensor([1, 4, 6])
one_hot = F.one_hot(ids, num_classes=vocab_size).float()
print(torch.allclose(linear(one_hot), embedding(ids)))  # True
```

这种等价关系解释了“Embedding 就是 one-hot 后接线性层”，但工程实现应优先使用查表，尤其是词表很大时。

## 10. Embedding 如何通过反向传播学习

若 batch 的 ID 为 `[3, 3, 8]`，第 3 行被查两次，第 8 行被查一次。反向传播时梯度会累积到被查询的行；未查询的行本 batch 梯度为零。

```python
embedding = nn.Embedding(10, 2)
ids = torch.tensor([3, 3, 8])
loss = embedding(ids).sum()
loss.backward()

print(embedding.weight.grad[3])  # 因为出现两次，梯度累加两次
print(embedding.weight.grad[8])  # 一次
print(embedding.weight.grad[1])  # 未出现，本 batch 梯度为 0
```

重复 ID 的梯度会按使用次数累计；padding 或大量重复 token 会改变每行收到的更新频率。

## 11. `padding_idx`：让 PAD 向量保持固定

文本 batch 常把短句补齐到统一长度。可设置 `padding_idx`，使该行不参与梯度更新，并在默认初始化下保持零向量：

```python
pad_id = 0
embedding = nn.Embedding(10, 4, padding_idx=pad_id)
ids = torch.tensor([[2, 3, 0], [4, 0, 0]])
vectors = embedding(ids)
print(vectors[ids == pad_id])

vectors.sum().backward()
print(embedding.weight.grad[pad_id])  # 零梯度
```

注意 `padding_idx` 不会自动让后续 attention 忽略 PAD，也不会自动让 loss 忽略 PAD。每种任务仍要分别构造 attention mask、loss `ignore_index` 或 masked pooling。

## 12. `max_norm`：查表时的行范数约束

`max_norm` 可限制每次 forward 中被访问的 embedding 行的范数：

```python
embedding = nn.Embedding(20, 8, max_norm=1.0)
ids = torch.tensor([3, 5, 3])
vectors = embedding(ids)
print(embedding.weight[3].norm().item() <= 1.0001)
```

它可能会原地修改被访问的权重行，因此若在同一 forward 前对 `embedding.weight` 做可微计算，需理解原地操作的 autograd 影响。不要把 `max_norm` 当作所有任务都需要的默认正则化。

## 13. `scale_grad_by_freq`：按 token 频率缩放梯度

`scale_grad_by_freq=True` 会按当前 mini-batch 内每个 ID 的频率缩小对应梯度，常用于高频类别行更新过强的情形：

```python
embedding = nn.Embedding(
    num_embeddings=100,
    embedding_dim=16,
    scale_grad_by_freq=True,
)
ids = torch.tensor([5, 5, 5, 7])
loss = embedding(ids).sum()
loss.backward()
print(embedding.weight.grad[5].norm(), embedding.weight.grad[7].norm())
```

这只影响梯度，不改变 forward 输出。使用前应与采样频率、loss reduction 和 optimizer 行为一起评估。

## 14. `sparse=True`：稀疏梯度与稀疏优化器

大词表 embedding 每个 batch 通常只访问少量行。`sparse=True` 可让梯度以稀疏形式存储，但只有部分 optimizer 支持稀疏梯度，例如 SGD、SparseAdam（支持范围依版本）：

```python
embedding = nn.Embedding(1_000_000, 64, sparse=True)
ids = torch.tensor([7, 10, 7, 500])
loss = embedding(ids).square().mean()
loss.backward()

optimizer = torch.optim.SparseAdam(embedding.parameters(), lr=1e-2)
optimizer.step()
```

不能想当然地将 sparse embedding 参数交给任意 optimizer（例如某些 Adam/AdamW 配置不接受 sparse gradient）。若模型同时有稠密参数，往往需要把参数分组并使用兼容的 optimizer。

## 15. Embedding 在文本模型中的位置

文本通常先经过 tokenizer 得到 token IDs，再由 embedding 层变成连续向量：

```text
原始文本 -> tokenizer -> [B,L] token IDs -> Embedding -> [B,L,D]
```

```python
vocab = {"<pad>": 0, "I": 1, "like": 2, "ML": 3}
token_ids = torch.tensor([[1, 2, 3, 0]])
embedding = nn.Embedding(len(vocab), 16, padding_idx=vocab["<pad>"])
token_vectors = embedding(token_ids)
print(token_vectors.shape)  # [1, 4, 16]
```

Embedding 不负责分词或把字符串映射成 ID；tokenizer/vocabulary 负责 ID 体系，且训练与推理必须使用兼容的 tokenizer 和 ID 映射。

## 16. 从 Embedding 到文本分类

最简单的句向量可通过对 token vectors 做平均池化得到，再接分类器：

```python
class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, token_ids):
        x = self.embedding(token_ids)            # [B,L,D]
        mask = token_ids.ne(self.pad_id).unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)            # [B,C]

model = MeanEmbeddingClassifier(vocab_size=1000, embedding_dim=32, num_classes=3)
token_ids = torch.tensor([[4, 8, 2, 0], [9, 5, 7, 3]])
logits = model(token_ids)
labels = torch.tensor([1, 2])
loss = F.cross_entropy(logits, labels)
print(logits.shape, loss.item())
```

必须 masked pooling；否则 PAD 的向量也会影响句子均值。这个模型是教学基线，不会建模词序，复杂文本可接 CNN、RNN 或 Transformer。

## 17. 类别特征 Embedding

表格数据的类别列常有高基数（例如城市、设备型号、用户 ID）。one-hot 可能很宽，可为每个类别学习低维向量：

```python
city_embedding = nn.Embedding(num_embeddings=5000, embedding_dim=16)
device_embedding = nn.Embedding(num_embeddings=300, embedding_dim=8)

city_ids = torch.tensor([10, 501, 10])
device_ids = torch.tensor([3, 7, 8])
features = torch.cat([city_embedding(city_ids), device_embedding(device_ids)], dim=-1)
print(features.shape)  # [3, 24]
```

类别 ID 应由训练集拟合的映射产生，并为未知类别预留 OOV/UNK ID；不要直接把任意字符串哈希到可能冲突的 ID 而不评估碰撞。

## 18. 推荐系统中的用户/商品 Embedding

矩阵分解的基本思想是给用户和商品各自学习向量，用户偏好可用内积打分：

$$
s(u,i)=p_u^Tq_i
$$

```python
num_users, num_items, dim = 1000, 5000, 32
user_emb = nn.Embedding(num_users, dim)
item_emb = nn.Embedding(num_items, dim)

users = torch.tensor([2, 2, 7, 9])
items = torch.tensor([10, 18, 10, 42])
scores = (user_emb(users) * item_emb(items)).sum(dim=-1)
print(scores.shape)  # [4]
```

用隐式反馈正负样本做二分类打分的训练片段：

```python
labels = torch.tensor([1.0, 1.0, 0.0, 1.0])
loss = F.binary_cross_entropy_with_logits(scores, labels)
loss.backward()
```

线上系统还要考虑新用户/新商品冷启动、时间/上下文特征、负采样偏差、热门度、召回与排序分层；单独的 embedding 内积只是基础模型。

## 19. 多字段类别特征拼接

多个离散字段可各用不同词表和 embedding dimension，拼接后与数值特征合并：

```python
class ProductFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.category = nn.Embedding(50, 8)
        self.brand = nn.Embedding(500, 12)
        self.region = nn.Embedding(20, 4)
        self.mlp = nn.Sequential(nn.Linear(8 + 12 + 4 + 3, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, category_id, brand_id, region_id, numeric):
        x = torch.cat([
            self.category(category_id),
            self.brand(brand_id),
            self.region(region_id),
            numeric,
        ], dim=-1)
        return self.mlp(x)

model = ProductFeatureModel()
score = model(
    torch.tensor([1, 2]),
    torch.tensor([20, 31]),
    torch.tensor([3, 4]),
    torch.randn(2, 3),
)
print(score.shape)  # [2, 1]
```

## 20. 共享 Embedding 与权重绑定

如果输入和输出共享同一词表，可考虑让输入 embedding 与输出分类矩阵共享权重（weight tying）。两者维度必须兼容：

```python
vocab_size, d_model = 1000, 64
token_embedding = nn.Embedding(vocab_size, d_model)
lm_head = nn.Linear(d_model, vocab_size, bias=False)
lm_head.weight = token_embedding.weight

ids = torch.randint(0, vocab_size, (2, 5))
hidden = token_embedding(ids)
logits = lm_head(hidden)
print(logits.shape)  # [2, 5, 1000]
print(lm_head.weight.data_ptr() == token_embedding.weight.data_ptr())  # True
```

权重绑定可减少参数并提供正则化效果，但只适合词表语义和维度匹配的场景。source 与 target 使用不同词表时，不应盲目共享。

## 21. 位置 Embedding 与 token Embedding 相加

Transformer 的 token embedding 表示“是什么”，position embedding 表示“在哪里”。两者通常具有相同维度并相加：

```python
vocab_size, max_length, d_model = 5000, 256, 64
token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
position_emb = nn.Embedding(max_length, d_model)

token_ids = torch.tensor([[5, 9, 2, 0], [4, 8, 7, 1]])
positions = torch.arange(token_ids.size(1)).unsqueeze(0)
x = token_emb(token_ids) + position_emb(positions)
print(x.shape)  # [2, 4, 64]
```

还可使用正弦位置编码、相对位置 bias 或 RoPE；这些提供位置信息的方式和词/类别 Embedding 表不是同一概念。

## 22. 预训练 Embedding

预训练词向量（如 Word2Vec、GloVe、FastText）在外部语料学习后可用作 embedding 初始化。关键是词表映射必须一致：

```python
vocab = {"<unk>": 0, "cat": 1, "dog": 2, "tree": 3}
pretrained = {
    "cat": torch.tensor([0.1, 0.2, 0.3]),
    "dog": torch.tensor([0.0, 0.2, 0.4]),
}

weight = torch.randn(len(vocab), 3) * 0.02
for word, idx in vocab.items():
    if word in pretrained:
        weight[idx] = pretrained[word]

embedding = nn.Embedding.from_pretrained(weight, freeze=False)
print(embedding(torch.tensor([1, 2, 3])).shape)
```

`freeze=False` 表示微调；`freeze=True` 表示训练时不更新表。缺失词向量应有明确定义，例如随机小值、UNK 向量或子词模型，而非悄悄使用错误行。

## 23. 冻结、部分冻结与微调

全部冻结：

```python
embedding = nn.Embedding.from_pretrained(weight, freeze=True)
print(embedding.weight.requires_grad)  # False
```

全部微调：

```python
embedding = nn.Embedding.from_pretrained(weight, freeze=False)
```

若只希望训练特定行，通常需要更明确的梯度 mask、拆分参数或定制 optimizer；简单地对参数整体设置 `requires_grad` 不能冻结单行。

```python
embedding = nn.Embedding(100, 16)
embedding.weight.requires_grad_(False)  # 全部冻结
```

## 24. Embedding 初始化与尺度

随机初始化尺度会影响早期激活和梯度。可显式初始化：

```python
embedding = nn.Embedding(10000, 128)
nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

with torch.no_grad():
    embedding.weight[0].zero_()  # 若 ID 0 是 padding，可设为零
```

PyTorch 的 `padding_idx` 应用时会让该行不参与梯度；若自己重新初始化或加载外部权重，也应检查 padding 行是否符合预期。

## 25. Embedding 的容量与参数量

参数量为 `V * D`，内存约为：

$$
\text{bytes}=V\times D\times\text{bytes per element}
$$

```python
V, D = 2_000_000, 128
params = V * D
fp32_gib = params * 4 / 1024**3
fp16_gib = params * 2 / 1024**3
print(f"parameters={params:,}, fp32={fp32_gib:.2f} GiB, fp16={fp16_gib:.2f} GiB")
```

训练时还需考虑梯度、optimizer 状态、checkpoint 副本与激活，内存远大于单份权重。超大 embedding 表可使用稀疏更新、分片、量化或托管 embedding 服务。

## 26. Embedding 学到的“相似”依赖目标函数

`nn.Embedding` 本身只提供参数查表，不会自动让语义相近的 ID 靠近。向量关系由监督任务决定：

- 分类目标让向量适合区分类别；
- 推荐目标让共现用户/商品关系反映偏好；
- 对比学习让正样本靠近、负样本分开；
- 语言模型让表示适合预测上下文 token。

一个简化的对比损失思想：

```python
z_a = F.normalize(torch.randn(8, 32, requires_grad=True), dim=-1)
z_b = F.normalize(torch.randn(8, 32, requires_grad=True), dim=-1)
similarity = z_a @ z_b.T / 0.1
labels = torch.arange(8)
loss = F.cross_entropy(similarity, labels)  # 对角线是匹配 pair
print(loss.item())
```

这只是结构示例，真实对比学习需处理数据增强、负样本、温度参数、分布式 batch 和评估协议。

## 27. Embedding 的相似度度量

向量检索常见度量：

| 度量 | 定义 | 特点 |
| --- | --- | --- |
| 内积 | `q @ x` | 受向量长度影响；常用于最大内积检索 |
| cosine similarity | `q·x/(||q||||x||)` | 只比较方向，需处理零向量 |
| Euclidean distance | `||q-x||_2` | 越小越近，受尺度影响 |

```python
q = torch.tensor([[1.0, 0.0]])
candidates = torch.tensor([[0.9, 0.1], [10.0, 0.0], [0.0, 1.0]])

dot = q @ candidates.T
cosine = F.normalize(q, dim=-1) @ F.normalize(candidates, dim=-1).T
distance = torch.cdist(q, candidates)
print("dot", dot)
print("cosine", cosine)
print("L2", distance)
```

内积更偏爱模长大的向量；cosine 忽略模长。选哪个必须与 embedding 模型训练目标和索引配置一致。

## 28. Top-k 向量近邻检索

小规模数据可直接暴力计算相似度：

```python
document_vectors = F.normalize(torch.randn(100, 32), dim=-1)  # [N,D]
query_vector = F.normalize(torch.randn(1, 32), dim=-1)       # [1,D]

scores = query_vector @ document_vectors.T                   # [1,N]
top_scores, top_indices = scores.topk(k=5, dim=-1)
print(top_indices.shape, top_scores.shape)  # [1,5]，[1,5]
```

暴力检索的复杂度约为 `O(ND)` 每条 query，结果精确但数据规模大时慢。生产系统通常建立 ANN（Approximate Nearest Neighbor，近似近邻）索引。

## 29. 归一化后的内积与 cosine 等价

对所有向量做 L2 normalization 后，内积等于 cosine similarity：

```python
vectors = torch.randn(10, 8)
normalized = F.normalize(vectors, p=2, dim=-1)
cosine = normalized @ normalized.T
print(torch.allclose(cosine.diag(), torch.ones(10), atol=1e-5))
```

若数据库用 inner product 索引实现 cosine 检索，必须对入库向量与查询向量都做相同归一化。只归一化一侧会改变排序。

## 30. 从文本到向量：Embedding Model 与向量库分工

语义检索通常不是把 token 的 `nn.Embedding` 行直接存库，而是使用句子/段落 embedding 模型把整段文本编码为固定维向量：

```text
文本 -> tokenizer/embedding model -> 向量 -> 向量数据库索引
查询 -> 同一兼容模型          -> query vector -> top-k 检索
```

同一检索空间要求 query 与 document embedding 模型、版本、维度、预处理和归一化兼容。随机 `nn.Embedding` 权重没有语义检索能力，不能未经训练就拿来做 RAG 检索。

纯 PyTorch 的 document 向量 shape 示例：

```python
document_embeddings = torch.randn(1000, 768)  # 1000 段文档，每段 768 维
query_embedding = torch.randn(1, 768)
assert document_embeddings.shape[1] == query_embedding.shape[1]
```

## 31. 向量数据库是什么

向量数据库/向量索引服务通常管理：

- 向量及其 ID；
- 元数据（来源、时间、租户、权限、文档类型等）；
- 可选原文或外部文档指针；
- ANN 索引与相似度度量；
- upsert、delete、filter、top-k search 等操作。

它不同于 `nn.Embedding`：

| 对比 | `nn.Embedding` | 向量数据库 |
| --- | --- | --- |
| 存的是什么 | 可训练参数行 | 编码后的对象向量、ID 和元数据 |
| 查询输入 | 离散整数 ID | 连续 query vector |
| 主要目标 | 端到端模型训练 | 大规模相似度搜索和过滤 |
| 是否自动学语义 | 由下游训练目标更新权重 | 通常不训练 embedding 模型 |
| 常见场景 | NLP、推荐、类别特征 | 语义搜索、RAG、相似商品/图片 |

## 32. 纯 Python/NumPy 精确检索基线

先建立一个不依赖向量数据库的正确性基线。小数据集使用矩阵乘法精确计算 cosine top-k：

```python
import numpy as np

texts = ["PyTorch tensors", "Transformer attention", "database indexing"]
vectors = np.random.randn(len(texts), 4).astype("float32")
query = np.random.randn(1, 4).astype("float32")

vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
scores = (query @ vectors.T)[0]
order = np.argsort(-scores)[:2]
print([(texts[i], float(scores[i])) for i in order])
```

这是 exact search，用来验证数据管线、度量和预期排序；它不具备 ANN 的规模优势，也不提供持久化、过滤或并发管理。

## 33. FAISS：向量近邻索引示例

FAISS 是面向向量相似度搜索的开源库，不等于托管数据库。安装和 GPU 支持依平台而异。下面用 CPU `IndexFlatIP` 建立精确内积索引；先归一化即可实现 cosine 排序：

```python
# 可选依赖：pip install faiss-cpu
import faiss
import numpy as np

doc_vectors = np.random.randn(1000, 64).astype("float32")
query_vectors = np.random.randn(2, 64).astype("float32")
faiss.normalize_L2(doc_vectors)
faiss.normalize_L2(query_vectors)

index = faiss.IndexFlatIP(64)  # 精确 inner product
index.add(doc_vectors)
scores, ids = index.search(query_vectors, k=5)
print(ids.shape, scores.shape)  # [2,5]、[2,5]
```

大规模时可考虑 IVF、HNSW、PQ 等近似索引，常以召回率、延迟、内存占用之间的折衷换速度：

```python
dimension = 64
quantizer = faiss.IndexFlatIP(dimension)
ivf = faiss.IndexIVFFlat(quantizer, dimension, 32, faiss.METRIC_INNER_PRODUCT)
ivf.train(doc_vectors)  # 训练索引结构，不是训练文本 embedding 模型
ivf.add(doc_vectors)
ivf.nprobe = 8           # 查询时探索的倒排分区数
scores, ids = ivf.search(query_vectors, 5)
```

索引的 `train()` 是学习/构建量化或分区结构；它不会让文档向量自动获得语言语义。

## 34. Chroma：带 ID 与 metadata 的向量集合

Chroma 等向量数据库提供 collection、ID、metadata、持久化和 query 接口。示例需要额外安装依赖，embedding 可由外部模型产生：

```python
# 可选依赖：pip install chromadb
import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="tutorial_docs",
    metadata={"hnsw:space": "cosine"},
)

collection.upsert(
    ids=["doc-1", "doc-2"],
    documents=["PyTorch Embedding 查表原理", "向量检索和 RAG 基础"],
    embeddings=np.random.randn(2, 64).astype("float32").tolist(),
    metadatas=[{"source": "chapter1", "tenant": "team-a"},
               {"source": "chapter2", "tenant": "team-a"}],
)
```

这里的随机向量仅用于演示 API，检索结果不具语义。实际应使用与 query 端相同的 embedding model 编码 documents。

查询与 metadata filter：

```python
query = np.random.randn(1, 64).astype("float32").tolist()
result = collection.query(
    query_embeddings=query,
    n_results=2,
    where={"tenant": "team-a"},
    include=["documents", "metadatas", "distances"],
)
print(result["ids"])
print(result["distances"])
```

不同版本对 collection 配置、距离返回和 embedding function 的 API 可能变化；示例演示概念，使用时核对安装版本文档。

## 35. Metadata filtering 与权限边界

向量相似度不是访问控制。多租户或私有语料检索时，应在搜索阶段应用可靠的 metadata filter，并在服务端验证用户权限：

```python
result = collection.query(
    query_embeddings=query,
    n_results=10,
    where={"tenant": "authorized-tenant"},
)
```

不要先跨租户搜出结果，再仅在前端隐藏；也不要信任客户端传入的任意 tenant id。元数据索引策略和过滤能力需按所选数据库验证。

## 36. RAG 检索的数据流

RAG（Retrieval-Augmented Generation）一般拆成离线索引和在线查询两条路径：

```text
离线：文档 -> 清洗/切块 -> embedding model -> vectors + metadata -> index
在线：问题 -> query embedding -> top-k + filter -> 可选 rerank -> prompt/context -> LLM
```

简化伪代码：

```python
def index_documents(chunks, embedder, vector_store):
    vectors = embedder.encode(chunks)  # [N,D]
    vector_store.upsert(
        ids=[chunk.id for chunk in chunks],
        embeddings=vectors,
        metadatas=[chunk.metadata for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
    )

def retrieve(question, embedder, vector_store, top_k=5):
    query_vector = embedder.encode([question])  # [1,D]
    return vector_store.search(query_vector, top_k=top_k)
```

`embedder` 与 `vector_store` 是不同职责：前者把文本变成向量，后者存储、索引并返回相似记录。

## 37. 文档切块与向量质量

检索质量不只由 embedding 模型决定，也受 chunking 影响。块太大可能混合多个主题，太小则缺上下文。简单按字符切片示意：

```python
def split_text(text, chunk_size=500, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

parts = split_text("一段较长的文档内容……" * 100)
print(len(parts), [len(p) for p in parts[:3]])
```

生产切块应尽量尊重段落、标题、代码块、句子边界和 token 数限制；还应保留文档 ID、页码、章节和版本等 metadata，方便引用与更新。

## 38. 向量索引的更新、删除与一致性

文档更新时需要同步更新向量和原文 metadata：

```python
collection.upsert(
    ids=["doc-1-chunk-0"],
    documents=["更新后的文本片段"],
    embeddings=[new_embedding.tolist()],
    metadatas=[{"source": "manual-v2", "updated_at": "2026-11-13"}],
)

collection.delete(ids=["obsolete-chunk-id"])
```

需要设计稳定 ID、重复 upsert 语义、删除策略、embedding model 版本标记和重建索引流程。文档文本变了但向量未重算会造成检索结果与展示内容不一致。

## 39. 检索评估：不要只看“结果看起来合理”

给定有标注的 query 和相关文档，可评估 Recall@K、MRR、nDCG 等检索指标：

```python
def recall_at_k(retrieved_ids, relevant_ids, k):
    hits = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(hits) / max(len(relevant_ids), 1)

retrieved = ["d3", "d1", "d8", "d2"]
relevant = {"d1", "d2"}
print(recall_at_k(retrieved, relevant, k=3))  # 0.5
```

区分两类质量：

- **Embedding 检索质量**：相关 chunk 是否进入 top-k。
- **ANN 索引召回**：近似索引相对 exact search 找回了多少真实近邻。

可以先用 exact search 建基线，再测 ANN 的 recall/latency/memory；否则很难判断问题来自 embedding 模型还是索引近似。

## 40. 常见错误与排查

| 现象 | 常见原因 | 检查方式 |
| --- | --- | --- |
| `index out of range` | ID 小于 0 或大于等于 `num_embeddings` | 检查 tokenizer/vocab 映射范围 |
| dtype 报错 | ID 是 float 或不兼容整型 | 使用 `torch.long` |
| PAD 影响句向量 | 只设 `padding_idx`，但未 masked pooling | 检查 pooling mask |
| PAD 仍产生 loss | 未设置 loss `ignore_index` | 单独屏蔽 padding 标签 |
| 所有相似度几乎相同 | 随机/未训练 embedding，模型空间不匹配 | 检查 encoder、模型版本和数据 |
| cosine 检索排序错 | query/doc 归一化不一致或度量设置错 | 两端同样 normalize，核对 metric |
| 数据库维数错误 | collection/index 维度与模型输出不一致 | 检查 `vectors.shape[-1]` |
| 更新后结果陈旧 | upsert/delete 或缓存未同步 | 检查向量 ID、版本、索引刷新 |
| RAG 召回错租户内容 | 搜索过滤/授权逻辑不严 | 服务端强制 metadata filter |

shape 和 ID 的快速断言：

```python
ids = torch.tensor([[1, 3, 0]], dtype=torch.long)
embedding = nn.Embedding(10, 8, padding_idx=0)
vectors = embedding(ids)

assert ids.dtype == torch.long
assert ids.min() >= 0 and ids.max() < embedding.num_embeddings
assert vectors.shape == (*ids.shape, embedding.embedding_dim)
assert torch.isfinite(vectors).all()
print("embedding checks passed")
```

## 41. 隐私、版本与生产注意事项

- 文本向量可能保留敏感信息或可被推断，不应视为天然匿名。
- 写入向量库前检查 PII、租户隔离、访问控制、保留期和删除要求。
- 记录 embedding model 名称、权重版本、维度、预处理和归一化配置。
- 更换 embedding 模型后，新旧向量通常处于不同空间，不应直接混合检索；应批量重算或隔离 collection。
- 保存原始对象 ID 和来源，支持追溯、去重、更新、删除和引用。
- ANN 参数调优要基于业务数据与延迟/召回目标，而非只看数据库默认配置。

## 42. 从零到一练习路线

1. 手写 one-hot，并验证它乘权重矩阵等于 embedding 查表。
2. 创建 `nn.Embedding`，检查权重表、输入输出 shape 和 ID 范围。
3. 用小分类任务训练 embedding，观察梯度只落在访问过的行。
4. 加入 PAD，分别实现 `padding_idx`、attention mask 和 masked pooling。
5. 做类别特征或用户/商品矩阵分解模型。
6. 加载预训练 embedding，比较冻结与微调效果。
7. 用 PyTorch/NumPy 实现 exact cosine top-k，建立检索基准。
8. 使用 FAISS 建立 Flat 与 IVF 索引，比较延迟和 Recall@K。
9. 使用向量数据库保存 ID、文本、向量和 metadata，并实现过滤查询。
10. 完成小型 RAG：切块、编码、入库、检索、引用来源，再评估 Recall@K。

## 43. 总结

Embedding 的核心是一个可学习查找表：

```text
离散 ID -> 查表 E[id] -> 连续向量 -> 下游任务通过梯度塑造表示
```

向量检索的核心则是：

```text
对象/文本 -> embedding model -> vector -> index/database
query -> 同一兼容空间中的 query vector -> similarity search -> top-k 对象
```

必须牢记：

- ID 是索引，不是具有大小语义的连续数值。
- `nn.Embedding(V,D)` 参数量为 `V*D`，输出 shape 是输入 ID shape 末尾追加 `D`。
- `padding_idx` 只处理 embedding 行更新，不会自动 mask attention 或 loss。
- Embedding 的语义来自训练目标/预训练，不是查表操作自动产生的。
- 向量数据库不会替你训练语义 encoder；检索质量取决于 embedding 模型、数据切块、度量和索引。
- `nn.Embedding` 适合模型训练中的离散特征；向量数据库适合保存并检索大量已编码对象；真实系统通常同时使用两者但用途不同。
