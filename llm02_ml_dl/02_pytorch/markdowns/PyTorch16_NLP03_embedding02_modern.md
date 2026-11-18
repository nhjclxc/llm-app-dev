# NLP 现代 Embedding：从 Word2Vec 到 BERT、Transformer 与 GPT

本文讲解 NLP 中“现代嵌入表示”的原理与实践。传统的 One-Hot、BoW、TF-IDF 和 N-gram 见配套的传统表示文档；这里重点讨论通过预测任务或上下文学习得到的稠密向量：Word2Vec 静态词向量、Transformer 的 token 表示、BERT 双向上下文表示、GPT 因果上下文表示，以及面向语义检索的句向量模型。

学习路线由浅到深：先理解静态与上下文向量的差别，再手工训练 Word2Vec 风格模型，然后拆解 Transformer/BERT/GPT 的向量从哪里来，最后用 Sentence Transformers 做句向量检索、微调与评估。每个代码片段均可独立理解；涉及 Hugging Face 预训练模型的片段需要安装依赖并联网下载模型权重。

## 1. 学习路线、术语与约定

| 阶段 | 问题 | 内容 |
| --- | --- | --- |
| 基础 | 什么是现代 embedding？静态和上下文向量差在哪？ | 2-3 |
| 入门实践 | Word2Vec 怎样从上下文学习词向量？ | 4-7 |
| Transformer | token、位置、层输出如何组成表示？ | 8-10 |
| 预训练模型 | BERT 和 GPT 如何产生不同的 contextual embedding？ | 11-14 |
| 应用 | 如何把 token 表示变成句向量并做检索？ | 15-18 |
| 进阶 | 微调、池化、评估、效率与常见陷阱 | 19-24 |

本文中的“embedding”可能指三种不同对象，阅读时要辨别：

1. **Embedding 参数表**：例如词表 ID 通过 Embedding 层查到的可训练向量。
2. **上下文 token 表示**：token embedding 经过 Transformer 多层计算后的 hidden state；同一个 token 放在不同句子中通常得到不同向量。
3. **句子/文档向量**：对 token 表示池化或由专用模型直接产生的固定长度向量，用于分类、聚类、语义搜索等。

统一用形状符号：B 为批大小，L 为 token 序列长度，D 为隐藏维度，V 为词表大小。很多张量以 [B, L, D] 表示；模型输出具体字段与维度要以所用模型配置为准。

## 2. 从稀疏表示到可学习的稠密表示

词是离散符号，模型需要数值输入。One-Hot 为每个词分配一个长度 V 的稀疏向量；它没有表达词义相近程度。现代方法把每个词或 token 映射为较低维的浮点向量，例如 D=300 或 D=768，并通过语言建模目标让有用的统计关系进入向量空间。

注意：向量“有语义”不是因为坐标天然代表某种人类概念，而是因为训练目标、数据和模型结构共同塑造了几何关系。余弦相似不必然等于同义，具体可用性需要按任务评估。

~~~python
import torch
from torch import nn

torch.manual_seed(7)
vocab = {"猫": 0, "狗": 1, "追": 2, "球": 3}
embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=3)
ids = torch.tensor([vocab["猫"], vocab["狗"]])
vectors = embedding(ids)
print(vectors.shape)  # [2, 3]
print(embedding.weight.shape)  # [词表大小, 向量维度]
~~~

这仍是一个可训练参数表：初始化时向量通常是随机的，只有参与训练后才可能学到任务有用的结构。若只是拿随机 embedding 做相似度，结果没有可靠语义。

## 3. 静态词向量与上下文表示

静态词向量为每个词类型保存唯一向量。Word2Vec 学到的“苹果”向量无论出现在水果还是公司语境中，都相同。它轻量、直观、词级查表快，但无法表达一词多义，也难处理未见词（OOV）。

上下文表示先将句子切成 token，再让 token 通过注意力等网络读取周围 token。于是“苹果很好吃”和“苹果发布了新芯片”中，苹果对应位置的 hidden state 可以不同。这不意味着每个语境都拥有单独的参数向量；模型用共享参数动态计算表示。

~~~python
# 静态查表：相同词 ID 的向量恒定
import torch
from torch import nn

static_table = nn.Embedding(10, 4)
apple_id = torch.tensor([3])
print(torch.equal(static_table(apple_id), static_table(apple_id)))  # True

# 上下文表示由整段 token 输入共同计算；下节将用 Transformer 展示
~~~

一个实用区分：Word2Vec 的词向量通常适合作为经典 NLP 特征或初始化；BERT/GPT 的最后层 token hidden states 是随上下文变化的表示。两类向量不能仅凭“都是浮点向量”就互换使用。

## 4. Word2Vec 的核心直觉

Word2Vec 是用浅层神经网络学习静态词向量的一类方法，常见训练目标是 CBOW 和 Skip-gram。它不是通过人工标注词义，而是利用“上下文相似的词往往具有相似分布”这一分布式假设：如果两个词经常出现在相似邻居附近，训练目标会推动它们的向量产生相似的预测行为。

给定句子“我 喜欢 深度 学习”，窗口大小为 1 时，中心词“喜欢”的邻居是“我”和“深度”。Word2Vec 建立中心词与上下文词的预测任务：

- CBOW：根据上下文预测中心词。
- Skip-gram：根据中心词预测周边上下文，通常对低频词更友好，但正样本更多。

~~~python
tokens = ["我", "喜欢", "深度", "学习"]
window = 1
pairs = []
for center_pos, center in enumerate(tokens):
    left = max(0, center_pos - window)
    right = min(len(tokens), center_pos + window + 1)
    for context_pos in range(left, right):
        if context_pos != center_pos:
            pairs.append((center, tokens[context_pos]))
print(pairs)
# ('我', '喜欢'), ('喜欢', '我'), ('喜欢', '深度'), ...
~~~

窗口越大，模型捕获的上下文范围通常更宽；窗口越小，关系更偏局部句法/搭配。训练语料、分词方式、词频截断和随机种子都会影响结果。

## 5. CBOW：用上下文预测中心词

设中心词为 w_t，窗口中的上下文词为 C_t。CBOW 先查出上下文 embedding，再做平均或求和得到上下文向量 h，最后对词表中每个候选词计算分数，通过 softmax 预测中心词：

~~~text
上下文 token IDs -> Embedding 查表 -> 向量平均 h
                 -> Linear(h) -> softmax(词表 V) -> 中心词概率
~~~

下面是最小完整 CBOW 模型和一个训练样例。小语料只用于说明梯度如何流经 embedding，不足以产生可靠通用词向量。完整 softmax 的分类头规模与 V 成正比，大词表通常改用负采样或层次 softmax。

~~~python
import torch
from torch import nn

corpus = "我 喜欢 自然 语言 处理 我 喜欢 深度 学习".split()
vocab = {word: i for i, word in enumerate(sorted(set(corpus)))}
encoded = [vocab[word] for word in corpus]

def make_cbow_examples(ids, radius=2):
    examples = []
    # 只保留完整窗口，保证每个样本上下文长度相同，便于直接组成 Tensor。
    for pos in range(radius, len(ids) - radius):
        context = [ids[j] for j in range(pos-radius, pos+radius+1) if j != pos]
        examples.append((context, ids[pos]))
    return examples

class CBOW(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, contexts):  # contexts: [B, C]
        hidden = self.input_embedding(contexts).mean(dim=1)  # [B, D]
        return self.output(hidden)  # [B, V] logits

examples = make_cbow_examples(encoded)
contexts = torch.tensor([ctx for ctx, _ in examples])
targets = torch.tensor([target for _, target in examples])
model = CBOW(len(vocab), dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
for _ in range(100):
    logits = model(contexts)
    loss = nn.functional.cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("loss:", loss.item())
~~~

此实现要求上下文长度一致；边界处上下文较短的真实训练代码通常采用 padding+mask、动态窗口或按样本组织的变长批次。训练后的 input_embedding.weight 是一种 CBOW 词向量选择，output 层的权重也含有预测空间信息；具体取哪套向量取决于实现和下游验证。

## 6. Skip-gram 与负采样

Skip-gram 对每个中心词分别预测窗口内的上下文词。直接对 V 个词做 softmax，每个正样本都要算整个词表，词表很大时昂贵。Negative Sampling（负采样）将多分类改写为若干二分类：真实中心词-上下文词对是正样本，随机抽到的噪声词对是负样本。

对中心向量 v_w 与上下文向量 u_c，模型用 sigmoid(v_w · u_c) 估计二者是真实共现的概率。目标推动正例点积变大、负例点积变小。常见噪声分布按 unigram 词频的 3/4 次方采样，但采样策略会随实现而异。

~~~python
import torch
from torch import nn
import torch.nn.functional as F

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)

    def forward(self, center_ids, positive_ids, negative_ids):
        # center_ids, positive_ids: [B]; negative_ids: [B, K]
        v = self.center(center_ids)                 # [B, D]
        u_pos = self.context(positive_ids)           # [B, D]
        u_neg = self.context(negative_ids)           # [B, K, D]
        pos_score = (v * u_pos).sum(-1)              # [B]
        neg_score = torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)  # [B, K]
        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).sum(1).mean()
        return loss

model = SkipGramNegativeSampling(vocab_size=100, dim=32)
center_ids = torch.tensor([4, 9, 3])
positive_ids = torch.tensor([5, 8, 2])
negative_ids = torch.randint(0, 100, (3, 5))  # 示范用均匀采样；真实训练应按词频采样
loss = model(center_ids, positive_ids, negative_ids)
loss.backward()
print(loss.item())
~~~

实际负采样要避免把正上下文误当负例，并合理构造词频噪声分布。这里刻意省略完整语料管线和 subsampling 高频词处理，重点展示目标函数；生产级训练建议使用成熟实现并检查其词表、采样和向量导出约定。

## 7. 词向量相似度与 Word2Vec 的边界

常用余弦相似度比较方向，而不是向量长度。相似度高只是当前向量空间中的邻近，不是逻辑蕴含、因果关系或严格同义关系。语料偏见、词频、领域变化都会反映到向量中。

~~~python
import torch
import torch.nn.functional as F

vectors = torch.randn(5, 8)  # 假设每行对应一个词
normalized = F.normalize(vectors, dim=-1)
cosine_matrix = normalized @ normalized.T
print(cosine_matrix.shape)  # [5, 5]
~~~

经典 Word2Vec 的主要边界：每个词通常只有一个向量；OOV 无向量；多义词混在一起；跨句信息有限。子词方法和上下文模型分别缓解其中部分问题，但模型变大并不自动保证下游表示更合适。

## 8. Transformer 表示由哪些部分构成

Transformer 通常先把 token ID 查成 token embedding，再加入位置表示，让模型区分顺序，之后通过多层 self-attention 和前馈网络更新每个位置的 hidden state。概念上：

~~~text
token IDs -> token embedding + position information
          -> Transformer Block x N
          -> contextual hidden states [B, L, D]
~~~

Token embedding 是输入层查表；位置编码可以是可学习向量、正弦位置编码或 RoPE 等方案；Transformer 输出的 hidden state 是经过上下文交互后的表示。不同架构和模型对 embedding 的实现各异，不能假设它们都是“词向量表”。

~~~python
import torch
from torch import nn

batch, length, vocab_size, dim = 2, 5, 100, 16
token_embedding = nn.Embedding(vocab_size, dim)
position_embedding = nn.Embedding(128, dim)  # 教学示例：可学习绝对位置向量
token_ids = torch.randint(0, vocab_size, (batch, length))
positions = torch.arange(length).unsqueeze(0).expand(batch, -1)
x = token_embedding(token_ids) + position_embedding(positions)
print(x.shape)  # [B, L, D]，尚未经过上下文层
~~~

位置表示告诉模型“谁在第几个位置”，而 self-attention 让一个位置按权重读取其他位置。具体 Q/K/V、mask、多头和残差等细节见 Attention 教程；本篇重点跟踪表示的来源和使用方式。

## 9. 为什么 Transformer 输出会随上下文变化

自注意力根据同一序列中 token 的内容和位置计算注意力权重，再将其他位置的信息混合进当前 token。经过多个 block 后，每个位置的 hidden state 不再只代表原始 token，而携带相关上下文。下方用 PyTorch 内置 TransformerEncoder 生成一个随机初始化的示范输出；没有训练时，其数值没有语言语义，只验证张量流与形状。

~~~python
import torch
from torch import nn

torch.manual_seed(0)
vocab_size, dim = 30, 16
embedding = nn.Embedding(vocab_size, dim)
layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=2)
ids = torch.tensor([[1, 2, 3, 4]])
mask = torch.zeros_like(ids, dtype=torch.bool)  # False 表示非 padding
input_vectors = embedding(ids)
hidden = encoder(input_vectors, src_key_padding_mask=mask)
print(hidden.shape)  # [1, 4, 16]
~~~

该简化示例未加位置编码、未训练，而且没有因果 mask；它是结构演示，不是 BERT 或 GPT 的等价实现。真实模型还包含预训练权重、特定 tokenizer、位置机制、归一化、dropout 和架构约定。

## 10. Hidden state、层选择与表示含义

模型的第 0 层常指输入 embedding，之后每一层产生新的 hidden state。浅层、中层、末层捕获的信息可能不同；下游任务不一定总是末层最好。Hugging Face 输出通常可通过 output_hidden_states=True 请求各层状态，但会增加显存占用。

~~~python
import torch
from transformers import AutoTokenizer, AutoModel

# 输出字段名因模型类/Transformers 版本而有差异，以下是 AutoModel 常见用法。
name = "bert-base-uncased"  # 示例；首次使用通常需要联网下载
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name, output_hidden_states=True)
inputs = tokenizer("A small example.", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # [B, L, D]
print(len(outputs.hidden_states))       # embedding 输出 + 各层输出（常见约定）
~~~

通用建议：先明确要取 token 级、句子级还是文档级表示；再确定模型层和池化方式；最后在目标数据上评估。只取某一层或把所有层平均并没有普适正确答案。

## 11. BERT 的双向上下文表示

BERT 是 Encoder-only Transformer。预训练时通过 Masked Language Modeling（MLM）遮住输入中部分 token，并预测原 token；经典 BERT 还使用过 NSP 目标。由于双向 self-attention，预测某位置时可利用左侧和右侧上下文。BERT 输出每个输入 token 的 contextual hidden state，常见维度由模型配置决定，例如 base 模型通常 D=768。

~~~python
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-chinese"  # 中文示例，需联网下载权重
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
text = "苹果发布了新产品。"
batch = tokenizer(text, return_tensors="pt", truncation=True)
with torch.no_grad():
    output = model(**batch)
token_vectors = output.last_hidden_state
print(batch["input_ids"].shape)
print(token_vectors.shape)  # [B, L, D]
~~~

Tokenizer 会把文本映射成模型训练时约定的 token ID，并加入特殊 token。必须使用与 checkpoint 匹配的 tokenizer；自行换分词器会导致 ID 语义错位，即使形状看起来正常也没有意义。

## 12. BERT 的 CLS、SEP 与句向量池化

CLS、SEP 是输入序列中的特殊 token，不是自然文本词语。CLS 位置的输出常被用于分类任务：将分类头接在 CLS 表示上并通过监督微调训练。直接把任意预训练 BERT 的 CLS 拿来做通用句向量，并不总能得到良好的语义相似度空间；原始 MLM 目标并未直接优化句向量余弦相似度。

Mean pooling 是另一种常见办法：对有效 token 表示求平均，忽略 padding；特殊 token 是否包含取决于模型和任务约定。下面展示正确使用 attention mask 排除 padding 的写法：

~~~python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)
texts = ["我喜欢自然语言处理", "自然语言处理很有趣"]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    output = model(**batch)
token_vectors = output.last_hidden_state
mask = batch["attention_mask"].unsqueeze(-1).to(token_vectors.dtype)
sentence_vectors = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
sentence_vectors = F.normalize(sentence_vectors, p=2, dim=-1)
print(sentence_vectors.shape)  # [B, D]
~~~

attention_mask 通常 1 表示有效 token、0 表示 padding；某些接口的其他 mask 约定需查文档。分母用 clamp 防止全 padding 导致除零。若要排除 CLS/SEP，应构造额外的 pooling mask，不能凭感觉把所有特殊 token 都删掉。

## 13. GPT 的因果上下文表示

GPT 是 Decoder-only Transformer，使用 causal mask 让位置 t 只能访问不晚于 t 的 token。自回归语言建模训练目标是根据前缀预测下一个 token：

~~~text
x1, x2, ..., xt -> 预测 x(t+1)
~~~

因此 GPT 位置 t 的 hidden state 表示“读完截至当前位置的前缀后，模型内部形成的状态”，它不能像 BERT 那样读取右侧未来文本。GPT 可用于生成，也能从 hidden state 构造特征，但取哪个位置、如何池化需与任务匹配。常见生成句向量做法取最后一个有效 token 的状态；不是所有 GPT checkpoint 都为语义相似度训练过。

~~~python
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

name = "distilgpt2"  # 演示模型，需联网下载；英文模型
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token  # 仅当该 tokenizer 没有 pad token 时的一种批处理设定
texts = ["A short sentence.", "A longer example sentence."]
batch = tokenizer(texts, padding=True, return_tensors="pt")
with torch.no_grad():
    out = model(**batch)
hidden = out.last_hidden_state
last_positions = batch["attention_mask"].sum(dim=1) - 1
sentence_vectors = hidden[torch.arange(hidden.size(0)), last_positions]
sentence_vectors = F.normalize(sentence_vectors, dim=-1)
print(sentence_vectors.shape)
~~~

左 padding 与右 padding 会改变“最后有效位置”的索引方式；上例使用默认右 padding，并以 attention_mask 计算位置。GPT 的 EOS 被复用作 PAD 是演示设定，不是所有 tokenizer/模型的通用最佳实践。

## 14. BERT 与 GPT 表示对照

| 维度 | BERT 类 Encoder | GPT 类 Decoder |
| --- | --- | --- |
| 注意力可见范围 | 通常双向，可看左右上下文 | 因果遮罩，只看当前位置及之前 |
| 典型预训练目标 | MLM 等去噪目标 | Next-token prediction |
| hidden state 含义 | token 在完整输入上下文中的表示 | token 在已读前缀下的状态 |
| 常见用途 | 分类、抽取、编码、理解 | 生成、续写，也可提取特征 |
| 句向量处理 | CLS 或 masked mean 等，需验证 | 常用最后有效 token 或专门池化，需验证 |

两类模型都可输出 [B,L,D] hidden state，但相同形状不代表相同语义空间。未经同一对比目标训练的 BERT 向量与 GPT 向量通常不能直接比较；不同 checkpoint 的向量空间也一般不能混合检索。

## 15. 为什么需要专门的 Sentence Embedding 模型

通用语言模型训练目标不等于“让语义相似句的向量接近”。Sentence-BERT 一类方法通过 Siamese/双塔结构分别编码句子，再用相似度、分类或对比损失训练句向量。对比学习常见做法是：同一语义的正样本对相互靠近，批次中其他样本作为负例或使用显式难负例。

这种训练直接优化向量空间，更适合语义搜索、去重、聚类和召回。选择模型要看语言、领域、文本长度、推理速度、授权、向量维度和评测集，而不只看参数规模。

~~~python
# 推荐用 sentence-transformers 封装好的模型，需额外安装 sentence-transformers 并联网下载模型。
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
sentences = ["如何学习自然语言处理？", "NLP 入门应该从哪里开始？"]
vectors = encoder.encode(sentences, normalize_embeddings=True, convert_to_tensor=True)
print(vectors.shape)  # [句子数, embedding_dim]
print(vectors[0] @ vectors[1])  # 已归一化时点积等于余弦相似度
~~~

模型名称只是示例，模型可用性、效果和许可会变化。生产使用前阅读模型卡，检查训练数据与使用限制，并在自己的查询/文档分布上评估。

## 16. 用句向量做语义相似度与检索

将查询和候选文本通过同一个编码器得到向量，再按余弦相似度或点积排序。向量必须由同一模型版本、同一预处理和同一 pooling 流程产生；索引中的文档向量和查询向量应共享兼容空间。

~~~python
import torch
import torch.nn.functional as F

documents = ["PyTorch 支持张量和自动求导", "北京今天有小雨", "Transformer 使用注意力机制"]
query = "什么是自注意力网络？"

# 演示函数：实际应用中将其替换为 sentence-transformers 等模型的 encode。
def encode(texts):
    torch.manual_seed(sum(map(ord, "|".join(texts))))
    return F.normalize(torch.randn(len(texts), 384), dim=-1)

doc_vectors = encode(documents)
query_vector = encode([query])
scores = query_vector @ doc_vectors.T
top_scores, top_indices = scores.topk(k=2, dim=-1)
for score, index in zip(top_scores[0], top_indices[0]):
    print(float(score), documents[int(index)])
~~~

上面 encode 使用随机数，仅展示检索代码的张量形状，不会产生真正语义排序。实际向量检索可以先用矩阵乘法做小规模精确搜索，再按规模选 FAISS 或向量数据库；检索质量还依赖切块、元数据过滤、重排和评测，embedding 不是完整 RAG 系统。

## 17. 批处理、padding 与 pooling 的正确性

不同文本长度必须组批时常 padding 到相同长度。模型 attention mask 通常用于阻止模型把 padding 当作有效输入；池化时也要用 mask 排除 padding。漏掉池化 mask 会让短文本向量被 padding 位置稀释，导致相似度不稳定。

~~~python
import torch

hidden = torch.randn(2, 4, 3)  # [B=2, L=4, D=3]
attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
print(mean.shape)  # [2, 3]
~~~

Mean pooling 不是万能：特殊 token 是否参与、prompt 是否剔除、长文本如何分块，都应遵循模型训练方式或实验证据。超长文本应检查模型最大输入长度；静默截断可能丢掉关键内容。

## 18. 对比学习的简化原理

对一个 batch 的 N 对正样本，分别得到 query 向量 q_i 和 positive 向量 p_i，归一化后计算相似度矩阵 S=QP^T。理想情况下每行对角线是正例，其他列是负例。用温度 τ 缩放相似度，再对角线标签做交叉熵，得到 InfoNCE 风格目标。温度控制 softmax 分布尖锐程度。

~~~python
import torch
import torch.nn.functional as F

batch_size, dim = 4, 8
queries = F.normalize(torch.randn(batch_size, dim), dim=-1)
positives = F.normalize(torch.randn(batch_size, dim), dim=-1)
temperature = 0.05
logits = queries @ positives.T / temperature  # [N, N]
labels = torch.arange(batch_size)              # 对角线为匹配正样本
loss = F.cross_entropy(logits, labels)
print(loss.item())
~~~

这个代码只展示损失结构，不代表训练数据已经存在真实正负样本。实际训练要防止 batch 内假负例、构造高质量正例、跨设备聚合负例，并确保数据切分没有相同文本泄漏。temperature、batch size、hard negative 策略都影响效果。

## 19. 微调 BERT 做分类：CLS 表示的任务化

预训练 hidden state 是通用表示。分类任务常在 CLS 输出上接线性层，通过标签监督共同更新分类头与编码器。若数据较少，可先冻结编码器或采用较低学习率；但冻结是否更好应由验证集决定。下面是模型结构片段，数据集与训练循环留给下游任务文档展开。

~~~python
from transformers import AutoModel
from torch import nn

class BertForClassification(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        cls_vector = output.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_vector))  # logits [B, num_labels]
~~~

某些架构的首 token 不叫 CLS，或模型建议不同 pooling；不要把 BERT 专用的首位约定硬套到所有模型。训练分类时使用 logits 输入 CrossEntropyLoss，不要先手动 softmax。

## 20. 领域微调与 LoRA 的取舍

当通用模型与目标领域差距明显时，可以继续预训练、监督微调或对比微调。全量微调更新所有参数，表达能力强但显存和存储成本高，也更容易在小数据上过拟合。LoRA 在部分线性层旁加入低秩可训练更新，冻结大部分基座参数，降低训练成本；它不自动提升效果，也必须保存基础模型标识和 adapter 配置才能复现。

~~~python
# 可选：需安装 peft；示例展示 LoRA 配置，具体 target_modules 需按模型结构确认。
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

base = AutoModel.from_pretrained("bert-base-chinese")
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
)
model = get_peft_model(base, lora_config)
model.print_trainable_parameters()
~~~

不同架构的模块命名可能是 q_proj/v_proj、query/value 或其他名称。先检查 model.named_modules()，并用一次前向和反向传播确认注入层可训练。PEFT 并未列入本文项目依赖，需单独安装。

## 21. 词向量迁移与模型版本一致性

现代 tokenizer 常把词拆成子词或字节片段。模型词表 ID 是 checkpoint 的一部分；不能将一个模型的 ID 直接送入另一个模型，也不能只复制 token embedding 而忽略 tokenizer 变化。保存向量时同时记录模型仓库/版本、tokenizer、归一化、池化、最大长度和预处理。

~~~python
from transformers import AutoTokenizer, AutoModel

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
text = "机器学习"
encoded = tokenizer(text, return_tensors="pt")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
print(encoded["input_ids"].shape)
~~~

中文模型也可能按字、词片段或子词切分，实际结果应通过 convert_ids_to_tokens 检查。词表大小、特殊 token 和截断策略都影响输入表示。

## 22. 表示评估：不要只看几条近邻

可按目标评估：分类看准确率/F1；检索看 Recall@K、MRR、nDCG；语义相似度看 Spearman/Pearson 相关；聚类看 NMI/ARI 或人工审阅。划分数据时避免重复/近重复内容跨集合泄漏。还应检查不同长度、领域、语言和负例难度下的分层表现。

~~~python
import torch
import torch.nn.functional as F

# 小例子：计算检索的 Recall@K。假设每个 query 的相关文档索引集合已知。
query_vectors = F.normalize(torch.randn(3, 8), dim=-1)
doc_vectors = F.normalize(torch.randn(10, 8), dim=-1)
relevant = [{2, 5}, {1}, {0, 7}]
scores = query_vectors @ doc_vectors.T
k = 3
retrieved = scores.topk(k, dim=-1).indices.tolist()
recall_at_k = sum(bool(set(row) & relevant[i]) for i, row in enumerate(retrieved)) / len(relevant)
print("Recall@K:", recall_at_k)
~~~

这个示例把“至少召回一条相关文档”的 query 比例作为简化 Recall@K；标准信息检索中的 Recall@K 也常按被召回相关文档数/全部相关文档数计算，要根据评测定义实现。报告指标时写清公式、候选集与过滤规则。

## 23. 推理效率与数值细节

批量编码比逐条调用更高效；推理使用 eval() 和 torch.inference_mode() 可关闭训练行为与梯度记录。GPU 上可使用合适精度，但应验证数值和检索排序。长序列 Transformer 的注意力计算成本随 L 增长较快，因此截断、分块、动态 padding 和批次长度分桶很重要。下面假设前文已创建 model 和 batch。

~~~python
import torch

model.eval()
with torch.inference_mode():
    output = model(**batch)

# 排序前通常以 float32 计算相似度，便于稳定比较；是否需要取决于规模和误差要求。
vectors = output.last_hidden_state.float()
~~~

不要为省显存而无条件把所有向量降精度；要比较 top-k 排序、召回指标和延迟。向量归一化应在库写入和查询时保持一致。

## 24. 常见错误与排查清单

1. **把随机初始化向量当语义向量**：检查是否加载预训练权重或完成有效训练。
2. **误把 token embedding 当上下文输出**：确认用的是模型最后/指定层 hidden state，而非输入查表结果。
3. **忘记 padding mask**：同时检查 Transformer 输入 mask 与 pooling mask。
4. **任意取 CLS 或平均池化**：查模型卡与训练目标，在目标数据评估多种策略。
5. **不同模型空间混用**：同一索引中的文档与查询必须来自兼容模型和处理流程。
6. **忽视截断和特殊 token**：打印 token 序列、长度、attention mask，确认内容未被意外裁剪。
7. **只看余弦相似度数值**：做成对检索评测，明确正负样本和 Recall@K 等指标。
8. **以为 GPT 可以看完整句右侧内容**：因果遮罩使每个位置只能读取前缀；与 BERT 的双向表示不同。

~~~python
import torch

# 快速检查批次的长度、padding 数量与输出形状；假设 model 和 batch 已创建。
print("input_ids:", batch["input_ids"].shape)
print("有效 token 数:", batch["attention_mask"].sum(dim=1))
with torch.inference_mode():
    out = model(**batch)
print("hidden:", out.last_hidden_state.shape)
~~~

## 25. 从 0 到 1 的实践顺序

建议按如下顺序建立知识：

1. 用 nn.Embedding 理解 ID 查表和向量维度。
2. 手工生成滑动窗口，训练小型 CBOW/Skip-gram，观察训练损失而不夸大语义效果。
3. 学会 Transformer 输入 embedding、位置表示、注意力和 [B,L,D] hidden state 的区别。
4. 使用匹配的 tokenizer 与 BERT/GPT checkpoint，打印 token、mask 和输出形状。
5. 对比 CLS、masked mean、最后有效 token 等池化方法，并理解 BERT/GPT 可见上下文差异。
6. 使用专门的句向量模型完成相似度检索，用业务数据构建验证集和 Recall@K 等指标。
7. 只有当基线有明确短板时，再尝试微调、难负例、LoRA、量化或向量索引优化。

现代 embedding 的核心不是“选一个维度更大的向量”，而是弄清楚：向量由什么目标训练、表示什么粒度、依赖哪些上下文、如何池化、和什么空间兼容，以及在目标任务上是否经过评估。
