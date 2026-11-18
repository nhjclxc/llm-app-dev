# NLP 自然语言处理：从文本表示到大语言模型

自然语言处理（Natural Language Processing，NLP）研究如何让计算机处理人类语言。文本看起来是字符串，但模型最终接收的是数字张量；因此 NLP 的完整问题不只是“选什么模型”，还包括如何定义任务、清理和切分文本、构造标签、表示序列、设计训练目标，以及正确评估结果。

本文是 NLP 学习总览，按从基础到进阶的顺序讲清概念与数据流。每个知识点都配有示例代码。更深入的专题会在后续子文档展开：Tokenizer 见 [PyTorch14_Tokenizer.md](PyTorch14_Tokenizer.md)，Embedding 见 [PyTorch15_Embedding.md](PyTorch15_Embedding.md)，RNN 见 [PyTorch10_RNN.md](PyTorch10_RNN.md)，Attention 见 [PyTorch13_Attention.md](PyTorch13_Attention.md)。代码以 Python、PyTorch 2.x 为主；部分传统机器学习示例需要安装 scikit-learn，预训练模型示例需要安装 transformers 并下载对应模型。

```python
import re
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__, "device:", device)
```

## 1. NLP 在解决什么问题

语言有词汇、语序、上下文、歧义和隐含知识。NLP 把这些语言现象转化为可定义、可学习、可评估的计算任务。相同文本可以对应不同任务：判断情感是分类，抽取人名是序列标注，回答问题是条件生成或抽取，翻译是序列到序列映射。

```python
examples = {
    "分类": ("这家餐厅很好吃", "正面"),
    "序列标注": ("张三在北京工作", ["人名", "介词", "地点", "动词"]),
    "生成": ("输入：Good morning. 目标语言：中文", "早上好。"),
}
for task, pair in examples.items():
    print(task, "输入=", pair[0], "目标=", pair[1])      
```

### 1.1 常见任务与输入输出

任务定义决定标签形式、模型输出层和损失函数。文本分类通常对整句输出一个类别；序列标注对每个 token 输出一个类别；语言模型预测下一个 token；翻译模型根据源序列逐步产生目标序列。不要先挑网络，再反过来猜标签应该是什么。

```python
task_shapes = {
    "情感分类": {"input_ids": ["B", "L"], "logits": ["B", "C"]},
    "词性标注": {"input_ids": ["B", "L"], "logits": ["B", "L", "C"]},
    "语言模型": {"input_ids": ["B", "L"], "logits": ["B", "L", "V"]},
}
for name, shapes in task_shapes.items():
    print(name, shapes)
```

符号含义：`B` 是 batch size，`L` 是 token 序列长度，`C` 是类别数，`V` 是词表大小。分类模型输出的 `logits` 是未归一化分数；通常直接交给 `CrossEntropyLoss`，不要预先做 softmax。

## 2. 文本不是数字：从字符串到样本

神经网络的输入是数值张量，不能直接把 Python 字符串送入 `nn.Linear`。首先需要规定文本单位（字符、词、子词或字节），再建立 token 到整数 ID 的映射。ID 只是索引，不自带词义，也没有大小关系。

```python
tokens = ["nlp", "models", "learn", "from", "text"]
vocab = {token: index for index, token in enumerate(tokens)}
sentence = ["models", "learn", "from", "text"]
input_ids = [vocab[token] for token in sentence]
print(input_ids)  # 整数序列；ID 的大小不代表语义   
```

预处理需要服务于具体任务。统一大小写可能帮助普通英文分类，但可能破坏专名或代码；去标点可能降低噪声，也可能删除情感符号、句界或结构信息。应把预处理规则视为模型的一部分，并在训练和推理时保持一致。

```python
text = "NLP is useful!"
normalized = text.lower().strip()
pieces = re.findall(r"\w+|[^\w\s]", normalized)
print(normalized, pieces)  # 教学用规则，不是通用生产 tokenizer
```

## 3. 词表、未知词和特殊 token

有限词表无法收录所有新词。传统词级方案通常用 `[UNK]`（unknown）表示未登录词；现代子词/字节级 tokenizer 会把罕见词拆成更小单位，降低未知词概率。批处理还需要 `[PAD]` 补齐长度，语言模型通常需要 `[BOS]`、`[EOS]` 表示序列边界，指令模型可能有角色或消息边界 token。

特殊 token 必须在 tokenizer 和模型中使用一致的 ID。它们不是“看起来特殊的字符串”就自动拥有特殊行为；其含义由词表、预处理器、模型训练约定共同决定。

```python
special = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
normal = {"hello": 4, "world": 5}
vocab = {**special, **normal}
text_tokens = ["[BOS]", "hello", "mystery", "[EOS]"]
ids = [vocab.get(token, vocab["[UNK]"]) for token in text_tokens]
print(ids)
```

## 4. Tokenizer：模型看到的文本单位

Tokenizer 将文本编码成 token ID 序列，也负责把 ID 解码回文本。字符级词表小但序列长；词级可读但词表大、容易遇到新词；子词级在词表规模与覆盖能力之间折中。BPE、WordPiece、Unigram 是不同的子词构造/切分方法，真实行为还取决于预切分、规范化和特殊 token 配置。

```python
text = "unbelievable"
char_tokens = list(text)
word_tokens = [text]
toy_subword_tokens = ["un", "believ", "able"]  # 仅作示意，非真实 tokenizer 输出
print(char_tokens)
print(word_tokens)
print(toy_subword_tokens)
```

子词切分的具体边界不是通用语言学真理。使用预训练模型时必须加载与权重配套的 tokenizer，不能自行建立一个“大小一样”的词表替代它。详细算法、截断、padding、mask、Chat Template 和代码实践见 Tokenizer 专题文档。

## 5. 文本分类的传统基线：词袋与 TF-IDF

词袋（Bag of Words）把文本表示成词项计数，忽略词序。它简单、快速、容易解释，是判断数据和标签是否可学的好基线。TF-IDF 会降低常见于许多文档的词的权重，突出更能区分文档的词。它们无法自然表示长距离语序或上下文含义。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

texts = ["good food and good service", "bad food and slow service"]
count = CountVectorizer()
X_count = count.fit_transform(texts)
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(texts)
print(count.get_feature_names_out())
print(X_count.toarray())
print(X_tfidf.toarray().round(2))
```

`fit` 只能在训练集上执行：它会学习词表和 IDF 权重。若先对全量数据 `fit_transform` 再拆训练/测试集，测试文本已经影响了特征空间，形成数据泄漏。测试集只允许 `transform`。

```python
from sklearn.model_selection import train_test_split

train_texts, test_texts = train_test_split(texts, test_size=0.5, random_state=42)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)  # 只在训练集 fit
X_test = vectorizer.transform(test_texts)         # 测试集复用训练词表和 IDF
print(X_train.shape, X_test.shape)
```

## 6. N-gram：有限地保留局部顺序

词袋无法区分 “dog bites man” 和 “man bites dog”。N-gram 把连续 `n` 个 token 作为特征，unigram 保留单词，bigram 额外保留相邻词对，trigram 保留相邻三词。它保留局部顺序，但特征数量会快速增长，也不能覆盖任意长距离关系。

```python
tokens = ["the", "model", "learns", "language"]
def ngrams(items, n):
    return [tuple(items[i:i + n]) for i in range(len(items) - n + 1)]

print(ngrams(tokens, 1))
print(ngrams(tokens, 2))
```

传统文本分类可直接用 `ngram_range` 加入 unigram/bigram，再接线性分类器；这经常是成本低、效果有竞争力的基线。

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = ["movie was wonderful", "movie was awful", "great acting", "terrible acting"]
labels = [1, 0, 1, 0]
baseline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000),
)
baseline.fit(texts, labels)
print(baseline.predict(["wonderful acting"]))
```

## 7. 语言模型：学习语言的概率结构

语言模型估计 token 序列的概率。利用链式法则，一句话的联合概率可分解为逐 token 条件概率：

$$
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1})
$$

自回归语言模型用前文预测下一个 token；掩码语言模型遮住部分 token，利用左右上下文恢复它们。前者天然适合逐步生成，后者擅长双向上下文表示。二者训练目标不同，不能只看模型名称判断其用途。

```python
# teacher forcing 的训练对：输入是前缀，目标向右移一位
ids = torch.tensor([[10, 11, 12, 13, 14]])
input_ids = ids[:, :-1]  # [10, 11, 12, 13]
target_ids = ids[:, 1:]  # [11, 12, 13, 14]
print(input_ids, target_ids)
```

若模型在位置 `t` 输出对词表的分数，就用该位置 logits 预测目标 token `t+1`。这种对齐关系是 causal language modeling 的核心，错一位仍可能运行，但学到的目标已经变了。

## 8. 文本向量化：One-hot 与 Embedding

One-hot 用长度为词表大小 `V` 的稀疏向量表示 token；Embedding 用 `D` 维可训练稠密向量表示 token。Embedding 可以看作查一个形状 `[V,D]` 的参数表。向量的语义结构来自训练目标，而不是整数 ID 本身。

```python
vocab_size, embedding_dim = 8, 4
ids = torch.tensor([[1, 3, 2]])
embedding = nn.Embedding(vocab_size, embedding_dim)
vectors = embedding(ids)
print(vectors.shape)  # [B=1, L=3, D=4]
```

分类任务常把 token 向量聚合成句向量。平均池化易理解，但应排除 padding；RNN/Transformer 则通过顺序或注意力构造上下文相关表示。Embedding 的训练、梯度、预训练向量和向量相似度详见 Embedding 专题。

```python
pad_id = 0
ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
embedding = nn.Embedding(6, 5, padding_idx=pad_id)
vectors = embedding(ids)                         # [B, L, D]
mask = ids.ne(pad_id).unsqueeze(-1)              # [B, L, 1]
sentence_vectors = (vectors * mask).sum(1) / mask.sum(1).clamp_min(1)
print(sentence_vectors.shape)                    # [B, D]
```

## 9. 序列长度：padding、mask 与截断

一个 batch 中不同文本长度不同，张量要求规则形状，因此通常把较短序列补成相同长度。Padding 是占位符，不是文本内容；若模型或池化把它当作真实 token，表示和 loss 会被污染。Attention mask 通常标出有效 token，但不同 API 对 mask 的真假含义可能相反，必须核对接口。

```python
pad_id = 0
sequences = [[4, 5, 6], [7, 8]]
max_len = max(map(len, sequences))
input_ids = torch.tensor([seq + [pad_id] * (max_len - len(seq)) for seq in sequences])
attention_mask = input_ids.ne(pad_id).long()
print(input_ids)       # [[4,5,6], [7,8,0]]
print(attention_mask)  # [[1,1,1], [1,1,0]]
```

超过模型最大长度的序列需要截断、分块或滑动窗口。截断可能丢掉关键信息；分块可能丢失块间上下文。训练、验证、推理的处理策略需要一致，并记录实际 token 长度分布。

```python
max_length = 4
ids = [2, 10, 11, 12, 13, 3]  # 假设 2/3 是边界标记
truncated = ids[:max_length]
print(truncated)
```

## 10. 从词频到监督分类：标签与损失

监督学习由输入 `x` 和目标 `y` 构成。二分类常用单个 logit 配 `BCEWithLogitsLoss`；多分类常用 `[B,C]` logits 配整数类别标签和 `CrossEntropyLoss`。后者内部会做 log-softmax，因此不要额外 softmax 后再传入。

```python
logits = torch.tensor([[1.2, -0.3, 0.1], [0.0, 0.8, -0.4]])  # [B=2, C=3]
labels = torch.tensor([0, 1])                                # class index, [B]
loss = nn.CrossEntropyLoss()(logits, labels)
predictions = logits.argmax(dim=-1)
print(loss.item(), predictions)
```

序列分类与 token 分类的 target 形状不同。Token 分类一般需要忽略 padding 标签，例如令 padding 位置标签为 `-100`（PyTorch `CrossEntropyLoss` 默认的 `ignore_index`），这样补齐位置不参与梯度和平均损失。

```python
batch, length, classes = 2, 4, 5
token_logits = torch.randn(batch, length, classes)
token_labels = torch.tensor([[1, 2, 0, -100], [3, 1, -100, -100]])
loss = nn.CrossEntropyLoss(ignore_index=-100)(
    token_logits.reshape(-1, classes), token_labels.reshape(-1)
)
print(loss.item())
```

## 11. 数据集切分与评估：测量泛化而不是记忆

训练集用于更新参数，验证集用于选模型和调超参数，测试集只在方案确定后用于最终评估。随机切分适合近似独立同分布数据；按时间预测未来时要按时间切分；同一用户、作者或近重复文本可能需要按 group 切分，避免相似样本跨集合泄漏。

```python
from sklearn.model_selection import train_test_split

texts = ["good film", "bad film", "great story", "poor story", "nice acting", "boring"]
labels = [1, 0, 1, 0, 1, 0]
train_x, test_x, train_y, test_y = train_test_split(
    texts, labels, test_size=0.33, random_state=42, stratify=labels
)
print(len(train_x), len(test_x))
```

Accuracy 在类别均衡时直观，但类别不平衡时可能误导。Precision 衡量预测为正的样本中有多少真为正；Recall 衡量真实正例中找回多少；F1 是两者的调和平均。排序/概率任务还可看 ROC-AUC、PR-AUC，并按业务代价选择阈值。

```python
from sklearn.metrics import classification_report, confusion_matrix

y_true = [1, 1, 1, 0, 0, 0]
y_pred = [1, 0, 1, 0, 1, 0]
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, zero_division=0))
```

## 12. PyTorch 训练循环的基本结构

训练循环的顺序是：切换训练模式、清零梯度、前向计算、计算损失、反向传播、更新参数。验证/推理切换到 `eval()` 并用 `torch.no_grad()`，从而关闭 Dropout 等训练行为并减少梯度开销。示例以已向量化的句向量为输入，聚焦训练机制。

```python
X = torch.randn(32, 20)                    # 32 个样本，每个 20 维
y = torch.randint(0, 2, (32,))
model = nn.Linear(20, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(X)
loss = criterion(logits, y)
loss.backward()
optimizer.step()
print("loss:", loss.item())
```

这个片段只走一个 batch。真实训练还要通过 `DataLoader` 遍历 batch，按 epoch 训练，验证集监控指标，保存最佳 checkpoint，并确认输入、标签、logits 的 shape 和 dtype。不要在测试集上反复选超参数。

## 13. 循环网络：RNN、LSTM 与 GRU

RNN 按时间顺序处理序列，用隐藏状态传递此前信息。它直观但长序列训练容易梯度消失/爆炸；LSTM 和 GRU 使用门控机制控制信息保留和更新。循环结构可以处理可变长度序列，但逐步计算限制并行度。

```python
batch, length, embedding_dim = 2, 5, 8
embedded = torch.randn(batch, length, embedding_dim)
rnn = nn.GRU(input_size=embedding_dim, hidden_size=16, batch_first=True)
outputs, final_state = rnn(embedded)
print(outputs.shape)      # [B, L, H]
print(final_state.shape)  # [层数, B, H]，单向单层时 [1, B, H]
```

RNN、双向序列、变长打包、隐藏状态和 BPTT 在 RNN 专题中深入讲解。本节只建立主线：文本 ID 先经 Embedding，循环层读入向量序列，最后按任务选择末状态或所有时间步的输出。

## 14. Attention 与 Transformer

Attention 让每个位置根据内容从其他位置读取信息。Self-attention 中 Query 与所有 Key 计算相关性，再对 Value 加权汇总；多头机制并行学习多种关系。Transformer 将自注意力、前馈网络、残差和归一化堆叠起来，较 RNN 更易并行建模长距离关系，但标准全注意力对长度 `L` 的计算/显存随 `L^2` 增长。

```python
batch, length, d_model = 2, 4, 16
x = torch.randn(batch, length, d_model)
layer = nn.TransformerEncoderLayer(
    d_model=d_model, nhead=4, batch_first=True
)
encoded = layer(x)
print(encoded.shape)  # [B, L, D]
```

Transformer 本身不天然知道 token 顺序，因此通常加入位置编码/位置嵌入。Encoder、causal decoder、cross-attention 和 padding/causal mask 的边界非常重要，详见 Attention 专题；文本 token 化和位置编码不可混为一件事。

```python
positions = torch.arange(length).unsqueeze(0).expand(batch, -1)
position_embedding = nn.Embedding(128, d_model)
x_with_position = x + position_embedding(positions)
print(x_with_position.shape)
```

## 15. 预训练、迁移学习与微调

大规模预训练先在大量无标注文本上学习通用表示或语言建模能力；微调再用较小的任务数据调整参数。冻结底层或只训练分类头成本低但适配能力有限；全量微调更灵活但需要更多显存、数据和正则化。参数高效微调（如 LoRA）只训练少量附加参数，减少存储与训练成本。

```python
hidden_size, num_classes = 16, 3
encoder = nn.Sequential(nn.Linear(8, hidden_size), nn.ReLU())  # 代替已加载的预训练编码器
for parameter in encoder.parameters():
    parameter.requires_grad = False  # 冻结编码器

classifier = nn.Linear(hidden_size, num_classes)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
```

使用 Hugging Face 时，模型、tokenizer 和配置应作为一套加载；以下伪代码展示调用形状，需把模型名和数据替换为真实可用资源。真实预训练权重可能需要联网下载。

```python
# pip install transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
batch = tokenizer(["great movie", "not good"], padding=True, truncation=True, return_tensors="pt")
outputs = model(**batch)
print(outputs.logits.shape)  # [B, 2]
```

## 16. 生成式 NLP：从条件概率到解码

生成模型每一步依据已有上下文产生下一个 token 的概率分布。Greedy decoding 选最高概率 token，简单但可能重复或缺乏多样性；temperature 调整分布尖锐程度；top-k/top-p 将采样限制在高概率候选中。解码策略不改变模型学到的参数，只改变生成时如何从分布中取样。

```python
logits = torch.tensor([2.0, 1.0, 0.2])
temperature = 0.7
probabilities = torch.softmax(logits / temperature, dim=-1)
next_id = torch.multinomial(probabilities, num_samples=1)
print(probabilities, next_id)
```

生成循环要把新 token 接回上下文，直到遇到 EOS、达到长度上限或满足停止条件。实际大模型还需 KV cache、采样过滤、重复惩罚、停止串等工程机制；对话模型还依赖正确的 chat template 和角色边界。

```python
next_token_logits = torch.tensor([0.1, 2.0, 0.4])
next_token_id = next_token_logits.argmax().item()  # greedy 的单步示意
generated_ids = [5, 8, next_token_id]
print(generated_ids)
```

## 17. 预训练目标与模型家族的差异

Encoder 类模型常通过 masked language modeling 学习双向上下文，适合理解、分类、抽取等表示任务；decoder-only 模型常通过 next-token prediction 训练，适合续写和指令生成；encoder-decoder 模型把输入编码后条件生成输出，常见于翻译、摘要等任务。这是常见倾向而非能力绝对边界，具体效果取决于训练数据、目标、规模和后训练。

```python
# MLM 的概念示意：仅用被遮住位置计算预测损失
vocab_size = 30
logits = torch.randn(1, 5, vocab_size)  # [B, L, V]
masked_targets = torch.tensor([[-100, -100, 12, -100, -100]])
loss = nn.CrossEntropyLoss(ignore_index=-100)(
    logits.reshape(-1, vocab_size), masked_targets.reshape(-1)
)
print(loss.item())
```

不同目标函数塑造不同训练信号。比较模型时应关注其训练目标和输入输出接口，而不是只根据“Transformer”或“LLM”标签推断适配性。

## 18. 数据质量、偏差与数据泄漏

模型能学到的数据规律受标注质量、样本覆盖和数据切分影响。重复文本跨越训练/测试集会虚高指标；训练时用了测试语料建立词表也属于泄漏；类别定义含糊会导致模型上限受标注一致性限制。应先检查样本、标签分布、重复和来源，再扩大模型。

```python
from collections import Counter

texts = ["positive review", "negative review", "positive review"]
labels = [1, 0, 1]
print("label counts:", Counter(labels))
print("duplicate texts:", len(texts) - len(set(texts)))
```

NLP 数据可能含个人信息、偏见、攻击性内容或受版权保护文本。实际项目应按用途控制数据采集与保留，脱敏并评估子群体误差；生成系统还需防范幻觉、提示注入和不适当输出。模型输出是概率性结果，不自动等于事实。

```python
# 一个最小的 PII 检查示意；生产环境需要经过评估的专用检测器
text = "联系我：alice@example.com"
redacted = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
print(redacted)
```

## 19. 常见问题与排查顺序

NLP 训练出错常常不是模型层本身，而是 tokenizer 不匹配、padding 参与池化/loss、target 错位、shape 广播或切分泄漏。先打印样本和张量契约，再检查模型容量与超参数。

```python
debug_input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
debug_labels = torch.tensor([0, 1], dtype=torch.long)
print("input_ids:", debug_input_ids.shape, debug_input_ids.dtype)
print("labels:", debug_labels.shape, debug_labels.dtype)
# 分类时通常要求 logits [B, C]、labels [B] 且 labels 为 long 类别索引
debug_logits = torch.randn(debug_labels.shape[0], 2)
assert debug_logits.ndim == 2 and debug_labels.ndim == 1
assert debug_labels.dtype == torch.long
```

| 表现 | 优先检查 |
| --- | --- |
| Loss 维度报错或数值异常 | logits/target shape、dtype、类别索引范围、是否错误广播 |
| 训练准确率高、测试很差 | 数据泄漏、重复样本、过拟合、分布漂移、切分方式 |
| 预测几乎只有一个类别 | 标签分布、类别权重、阈值、学习率、标签映射 |
| 同一句编码结果不一致 | tokenizer、规范化、特殊 token、模型版本是否一致 |
| 句向量随 padding 改变 | pooling 是否使用有效 token mask |
| 生成不断重复 | EOS/停止条件、解码策略、上下文长度和训练分布 |

## 20. 推荐学习路线与项目

学习顺序应让每一步都能解释数据如何变化：文本清理和统计特征建立基线；Tokenizer/Embedding 把离散文本转成张量；分类训练循环建立监督学习流程；再学习序列模型和 Attention；最后接触预训练与生成。不要跳过传统基线和评估，它们能帮助判断深度模型到底解决了什么问题。

```python
learning_path = [
    "词袋 / TF-IDF + Logistic Regression：情感分类基线",
    "Tokenizer + Embedding + 平均池化：PyTorch 文本分类",
    "GRU/LSTM：短文本分类或序列标注",
    "Transformer encoder：微调预训练模型",
    "自回归语言模型：理解 next-token 训练与解码",
]
for index, step in enumerate(learning_path, start=1):
    print(index, step)
```

建议实践项目：新闻主题分类、垃圾短信识别、中文命名实体识别、短文本意图分类、新闻标题生成。每个项目至少记录数据来源、标签定义、切分策略、tokenizer、模型配置、验证指标和错误案例。

## 21. 总结：贯穿 NLP 的数据流

一个 NLP 系统可以概括为：明确任务与标签，清理并切分文本，使用确定的 tokenizer 编码为 ID，构造 batch 和 mask，通过 Embedding/模型产生 logits，按与任务匹配的目标训练，再用独立数据评估并分析错误。任何一环的约定变化都可能改变模型输入或指标含义。

```text
原始文本
  -> 规范化与 tokenizer
  -> token IDs / attention mask / labels
  -> Embedding 与 RNN / CNN / Transformer
  -> logits 或生成分布
  -> loss 与优化
  -> 验证、测试、错误分析与部署
```

接下来的子文档可以分别深入 Tokenizer、Embedding、文本分类完整项目、RNN 序列建模、Attention/Transformer、预训练模型微调和生成式 NLP。理解这条数据流后，学习新架构时就能持续追问：输入表示是什么、模型输出什么、目标如何对齐、哪些位置参与损失、评估是否独立。
