# NLP 传统文本表示：从 One-Hot、BoW 到 TF-IDF 和 N-gram

Embedding（嵌入表示）不一定一开始就指神经网络中的 nn.Embedding。在自然语言处理的经典阶段，人们先把文本转换成可计算的向量，再交给相似度算法、线性模型或浅层神经网络。常见传统表示包括：

- One-Hot：用一个位置为 1、其余位置为 0 的稀疏向量表示 token；
- BoW（Bag of Words）：用词频表示一篇文档，忽略词序；
- N-gram：把连续的 n 个 token 作为特征，保留有限的局部顺序；
- TF-IDF：降低整个语料中普遍出现的词的权重，突出文档特有词；
- 哈希向量、共现矩阵和 LSA/LSI：进一步压缩或变换稀疏表示。

这些方法通常被称为传统文本向量化或传统 embedding。它们与神经网络中的稠密、可训练 embedding 有重要区别：传统方法往往由词频和统计公式直接构造，神经 embedding 则通过训练目标学习参数。本文从 0 到 1 讲清这条演进路线，并在最后比较它们与 PyTorch nn.Embedding、Transformer tokenizer 的边界。

本文示例主要使用 Python、PyTorch、NumPy、scikit-learn 和 jieba。项目中已经声明 NumPy、scikit-learn 和 jieba 依赖；若在独立环境运行，请先安装对应包。

```python
import re
import math
from collections import Counter

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)
```

## 1. 文本为什么需要向量表示

计算机模型不能直接对字符串做矩阵乘法。文本必须经过一条表示流程：

```text
原始字符串
    -> 规范化
    -> 分词
    -> 建立词表
    -> token / 文档转换成向量
    -> 相似度、分类、聚类或模型训练
```

对于一篇文档，向量的每个维度可以表示一个词、一个字符、一个 N-gram 或一个哈希桶。向量化的设计决定了模型能看到哪些信息。

```python
text = "machine learning is useful"
tokens = text.split()
vocabulary = sorted(set(tokens))
print(tokens)
print(vocabulary)
```

传统表示通常具有可解释性：某一维对应哪个词、某个词出现了多少次都能直接查看。但它们也可能维度很高、稀疏，并且难以表达长距离语义和上下文。

```python
documents = [
    "machine learning",
    "deep learning",
    "natural language processing",
]
vocabulary = sorted({word for doc in documents for word in doc.split()})
print("vocabulary size:", len(vocabulary))
print("feature meaning:", vocabulary)
```

## 2. Token、词表和文档

Token 是文本被切分后的基本单位，可以是字符、词、子词、标点或 N-gram。词表 vocabulary 是所有特征 token 的集合。将词表固定顺序后，才能让向量的每个位置有稳定含义。

```python
tokens = ["cat", "sat", "on", "mat"]
vocabulary = {token: index for index, token in enumerate(tokens)}
print(vocabulary)
print(vocabulary["cat"])
```

词表中的 ID 是索引，不表示大小关系，也不表示语义距离。ID 0 和 ID 1 只是两个列位置，不能直接作为连续特征送入线性模型。

```python
ids = torch.tensor([0, 1, 2])
linear = nn.Linear(1, 4)
print(linear(ids.float().reshape(-1, 1)))
```

如果训练集和测试集各自建立词表，列含义可能不同。正确做法是只用训练集拟合词表，再用同一个词表转换验证集和测试集。

```python
train_vocab = {word: i for i, word in enumerate(["cat", "dog", "runs"])}
test_tokens = ["dog", "bird"]
test_ids = [train_vocab.get(word, -1) for word in test_tokens]
print(test_ids)  # bird 不在训练词表中，-1 仅作演示占位，不能直接作为 Embedding ID
```

未知词可以被忽略、映射到 UNK，或通过字符/子词表示覆盖。无论选择哪种方案，都应在训练和推理阶段保持一致。

```python
UNK = "<UNK>"
vocab = {"cat": 0, "dog": 1, UNK: 2}
tokens = ["cat", "bird", "dog"]
ids = [vocab.get(token, vocab[UNK]) for token in tokens]
print(ids)
```

## 3. One-Hot 的原理

假设词表大小是 V，token 的 ID 是 i，则它的 One-Hot 向量长度为 V：

```text
token = vocabulary[i]
one_hot[i] = 1
其他位置 = 0
```

例如词表为 [cat, dog, fish, bird]，dog 的 ID 是 1，则：

```text
dog -> [0, 1, 0, 0]
```

```python
vocabulary = ["cat", "dog", "fish", "bird"]
token_to_id = {word: i for i, word in enumerate(vocabulary)}
token_id = token_to_id["dog"]

one_hot = F.one_hot(torch.tensor(token_id), num_classes=len(vocabulary))
print(one_hot)
print(one_hot.shape)
```

一批 token 可以一次性转换为二维矩阵，形状为 [N,V]。N 是 token 数，V 是词表大小。

```python
ids = torch.tensor([0, 1, 3])
one_hot = F.one_hot(ids, num_classes=4).float()
print(one_hot)
print(one_hot.shape)  # [N=3, V=4]
```

One-Hot 是离散对象的无歧义表示：不同 ID 的向量一定不同。但两个不同 token 的欧氏距离相同，不能表达词之间的语义相似。

```python
a = F.one_hot(torch.tensor(0), num_classes=4).float()
b = F.one_hot(torch.tensor(1), num_classes=4).float()
c = F.one_hot(torch.tensor(3), num_classes=4).float()
print(torch.dist(a, b))
print(torch.dist(a, c))
```

## 4. One-Hot 的优点与局限

One-Hot 的优点：

- 规则简单，容易解释；
- 每个 token 有独立且明确的维度；
- 不需要训练即可构造；
- 适合小词表和类别编码。

One-Hot 的局限：

- 词表大时向量维度很高；
- 大多数元素都是 0，计算和存储稀疏；
- 不表达语义相似；
- 不表达词序；
- 新词必须扩展词表。

```python
vocab_size = 100_000
one_hot = torch.zeros(vocab_size)
one_hot[12345] = 1
print(one_hot.numel(), one_hot.nonzero().shape)
```

上例中的向量只有一个非零元素，却分配了整个长度。真实文本词表可能达到几十万甚至更大，直接构造 dense One-Hot 不经济。

```python
ids = torch.tensor([10, 12345, 99999])
sparse_indices = torch.stack([ids, torch.arange(ids.numel())])
sparse_values = torch.ones(ids.numel())
sparse_one_hot = torch.sparse_coo_tensor(
    sparse_indices,
    sparse_values,
    size=(vocab_size, ids.numel()),
)
print(sparse_one_hot)
```

稀疏张量可以节省存储，但很多算子对稀疏格式的支持更复杂。实际神经网络通常使用 Embedding 查表，避免构造 One-Hot。

## 5. One-Hot 与线性层的关系

设 One-Hot 向量为 e_i，权重矩阵为 W。若线性层没有 bias：

```text
e_i @ W = W 的第 i 行
```

因此，One-Hot 后接 Linear 的结果与直接查一个 Embedding 表相同。

```python
vocab_size, dim = 5, 3
weight = torch.arange(vocab_size * dim, dtype=torch.float32).reshape(vocab_size, dim)
ids = torch.tensor([1, 3])

one_hot = F.one_hot(ids, num_classes=vocab_size).float()
via_linear = one_hot @ weight
via_lookup = weight[ids]

print(via_linear)
print(torch.equal(via_linear, via_lookup))
```

PyTorch 中可以用无 bias 的 Linear 模拟查表：

```python
embedding = nn.Embedding(vocab_size, dim)
linear = nn.Linear(vocab_size, dim, bias=False)

with torch.no_grad():
    linear.weight.copy_(embedding.weight.T)

ids = torch.tensor([0, 2, 4])
one_hot = F.one_hot(ids, num_classes=vocab_size).float()
print(torch.allclose(linear(one_hot), embedding(ids)))
```

这种等价关系解释了 Embedding 的本质，但工程上应优先使用查表，因为词表很大时 One-Hot 会浪费内存和计算。

## 6. 文档级 One-Hot 与词袋

One-Hot 表示一个 token。要表示一篇文档，可以把文档包含的 token 的 One-Hot 向量相加，得到词频向量，这就是 BoW 的基本思想。

```python
vocabulary = ["cat", "dog", "runs"]
token_to_id = {word: i for i, word in enumerate(vocabulary)}
document = ["cat", "cat", "runs"]

ids = [token_to_id[word] for word in document]
one_hot = F.one_hot(torch.tensor(ids), num_classes=len(vocabulary)).float()
bow = one_hot.sum(dim=0)

print(one_hot)
print(bow)  # cat 出现 2 次，runs 出现 1 次
```

BoW 的维度等于词表大小，维度 i 表示词表第 i 个词。它保留词频，不保留词出现的顺序和位置。

```python
doc_a = ["dog", "bites", "man"]
doc_b = ["man", "bites", "dog"]
vocabulary = sorted(set(doc_a + doc_b))
index = {word: i for i, word in enumerate(vocabulary)}

def bow_count(tokens, index):
    vector = torch.zeros(len(index))
    for token in tokens:
        vector[index[token]] += 1
    return vector

print(bow_count(doc_a, index))
print(bow_count(doc_b, index))
```

两个句子的 BoW 向量完全相同，但实际语义和主客体关系可能不同。这是 BoW 忽略词序的典型局限。

## 7. 手写 BoW 向量化器

手写一个简单向量化器有助于理解 CountVectorizer 的原理：先建立词表，再按词表顺序统计每篇文档中的词。

```python
class SimpleBoW:
    def __init__(self):
        self.vocabulary_ = {}

    def fit(self, documents):
        words = sorted({word for doc in documents for word in doc.split()})
        self.vocabulary_ = {word: i for i, word in enumerate(words)}
        return self

    def transform(self, documents):
        matrix = torch.zeros(len(documents), len(self.vocabulary_))
        for row, doc in enumerate(documents):
            for word, count in Counter(doc.split()).items():
                if word in self.vocabulary_:
                    matrix[row, self.vocabulary_[word]] = count
        return matrix

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)

documents = ["cat sat sat", "dog sat", "cat runs"]
vectorizer = SimpleBoW()
X = vectorizer.fit_transform(documents)
print(vectorizer.vocabulary_)
print(X)
```

transform 阶段只使用 fit 阶段得到的词表。测试集出现新词时，可以忽略或映射到专门的未知词维度。

```python
train_documents = ["red apple", "green apple"]
test_documents = ["blue apple"]

vectorizer = SimpleBoW().fit(train_documents)
print(vectorizer.transform(test_documents))
```

## 8. scikit-learn 的 CountVectorizer

scikit-learn 的 CountVectorizer 将原始文本转换成词频矩阵。它会建立词表、统计词频，并通常返回 scipy 稀疏矩阵。

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "cat sat on mat",
    "dog sat on log",
    "cat and dog",
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.shape)
print(X.toarray())
```

X 的行对应文档，列对应词表中的词，元素是该词在该文档中的次数。

```python
for document, row in zip(documents, X.toarray()):
    print(document, row)
```

默认 tokenizer 对英文按规则切分。中文没有空格，直接使用默认 CountVectorizer 往往不能得到想要的中文词粒度。

```python
documents = ["我喜欢自然语言处理", "自然语言处理很有趣"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
```

中文任务应先使用 jieba 切词并传入自定义 tokenizer，或提前把 token 用空格连接。

```python
import jieba

def chinese_tokens(text):
    return [word for word in jieba.lcut(text) if word.strip()]

vectorizer = CountVectorizer(
    tokenizer=chinese_tokens,
    token_pattern=None,
)
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

## 9. BoW 的二值化表示

词袋不一定只保存词频，也可以只表示词是否出现。二值 BoW 中出现为 1，不出现为 0，常用于文档分类和朴素贝叶斯等模型。

```python
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform([
    "cat cat dog",
    "dog fish",
])
print(X.toarray())
```

词频和二值出现具有不同含义：词频表示重复次数，二值表示存在性。重复出现是否代表更强证据，应该通过验证集比较，而不是预先假定。

```python
count_vectorizer = CountVectorizer(binary=False)
binary_vectorizer = CountVectorizer(binary=True)
count_X = count_vectorizer.fit_transform(["good good movie"])
binary_X = binary_vectorizer.fit_transform(["good good movie"])
print(count_X.toarray())
print(binary_X.toarray())
```

## 10. 文档长度与归一化

长文档天然包含更多词，原始计数可能偏爱长度更长的文档。可以使用 L1 或 L2 归一化，让向量更多表达词分布而非文档长度。

```python
from sklearn.preprocessing import normalize

counts = np.array([
    [2.0, 1.0, 0.0],
    [20.0, 10.0, 0.0],
])
print(normalize(counts, norm="l1"))
print(normalize(counts, norm="l2"))
```

归一化不会增加语义，只改变向量尺度。余弦相似度本身会忽略向量长度，而点积会受到长度影响。

```python
from sklearn.metrics.pairwise import cosine_similarity

x = np.array([[2.0, 1.0, 0.0]])
y = np.array([[20.0, 10.0, 0.0]])
print(cosine_similarity(x, y))
```

## 11. N-gram 的原理

N-gram 是连续 n 个 token 的片段：

- unigram：单个 token；
- bigram：连续两个 token；
- trigram：连续三个 token。

N-gram 在 BoW 基础上保留局部词序，可以区分“机器学习”和“学习机器”等不同局部结构。

```python
tokens = ["自然语言", "处理", "需要", "上下文"]

def make_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

print(make_ngrams(tokens, 1))
print(make_ngrams(tokens, 2))
print(make_ngrams(tokens, 3))
```

N-gram 数量随着词表大小和 n 增长，容易产生高维稀疏矩阵。实际使用通常限制 n 范围、设置最小文档频率或限制最大特征数。

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform([
    "machine learning is useful",
    "deep learning is useful",
])
print(vectorizer.get_feature_names_out())
print(X.shape)
```

## 12. N-gram 如何保留局部顺序

BoW 对“dog bites man”和“man bites dog”得到相同词频。加入 bigram 后，连续词对不同，向量也会不同。

```python
documents = ["dog bites man", "man bites dog"]
unigram = CountVectorizer(ngram_range=(1, 1))
bigram = CountVectorizer(ngram_range=(1, 2))

print(unigram.fit_transform(documents).toarray())
print(bigram.fit_transform(documents).toarray())
print(bigram.get_feature_names_out())
```

N-gram 只保留有限窗口内的顺序，不能理解远距离依赖，也不能自动知道同义词之间的关系。

```python
documents = [
    "the movie was not good",
    "the movie was good",
]
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
```

bigram 能看到“not good”这个局部组合，比删除所有停用词后只留下 good 更能保留否定信息。

## 13. 字符 N-gram

N-gram 不限于词，也可以在字符层面构造。字符 N-gram 对拼写变化、后缀、未登录词和混合语言有一定鲁棒性。

```python
vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
)
X = vectorizer.fit_transform(["playing", "played", "player"])
print(vectorizer.get_feature_names_out()[:20])
print(X.shape)
```

字符 N-gram 特征可能很多，但对拼写错误、词形变化和部分中文短文本常有帮助。词 N-gram 与字符 N-gram 可以分别训练或拼接。

```python
word_vectorizer = CountVectorizer(ngram_range=(1, 2))
char_vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 4))
word_X = word_vectorizer.fit_transform(["机器学习很好", "机器学习很有用"])
char_X = char_vectorizer.fit_transform(["机器学习很好", "机器学习很有用"])
print(word_X.shape, char_X.shape)
```

## 14. TF-IDF 的动机

BoW 只统计词频，但高频词不一定最能区分文档。例如“的”“是”“学习”可能在很多文档中出现。TF-IDF 结合两个因素：

- TF（Term Frequency）：词在当前文档中出现得多不多；
- IDF（Inverse Document Frequency）：词在整个文档集合中普遍不普遍。

直觉是：某词在当前文档多、在其他文档少，它更可能是该文档的代表性特征。

```text
TF-IDF(term, document)
    = TF(term, document) × IDF(term, corpus)
```

```python
documents = [
    "cat sat on mat",
    "cat ate fish",
    "dog ate fish",
]
word = "mat"
document_tokens = documents[0].split()
tf = document_tokens.count(word) / len(document_tokens)
df = sum(word in doc.split() for doc in documents)
idf = math.log((len(documents) + 1) / (df + 1)) + 1
print("tf:", tf, "df:", df, "idf:", idf, "tfidf:", tf * idf)
```

不同库和版本可能使用不同 TF、IDF 平滑和归一化公式。理解公式比死记某个数值更重要。

## 15. TF 的常见定义

TF 可以使用原始词频、文档内比例或对数缩放。原始词频容易让长文档占优势；比例会按文档长度归一；对数缩放可减弱重复词的边际影响。

```python
count = 5
document_length = 100
raw_tf = count
relative_tf = count / document_length
log_tf = 1 + math.log(count) if count > 0 else 0
print(raw_tf, relative_tf, log_tf)
```

这些公式都可能合理，选择取决于模型和数据。生产中通常使用经过验证的 TfidfVectorizer，并在实验记录中说明配置。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["cat cat dog", "dog fish"]
for sublinear in [False, True]:
    vectorizer = TfidfVectorizer(sublinear_tf=sublinear)
    X = vectorizer.fit_transform(documents)
    print(sublinear, X.toarray())
```

## 16. IDF 的直觉

如果一个词出现在几乎所有文档中，它的 IDF 较低；如果只出现在少数文档，它的 IDF 较高。IDF 不是词的固定属性，换一个语料集合，数值也会变化。

```python
documents = [
    "common rare_a",
    "common rare_b",
    "common rare_c",
]
N = len(documents)
for term in ["common", "rare_a"]:
    df = sum(term in doc.split() for doc in documents)
    idf = math.log((N + 1) / (df + 1)) + 1
    print(term, "df=", df, "idf=", idf)
```

IDF 必须只用训练语料拟合。若把测试文档也用于计算 IDF，测试分布会泄漏到训练特征中。

```python
train_documents = ["cat dog", "cat fish"]
test_documents = ["rare bird"]

vectorizer = TfidfVectorizer()
train_X = vectorizer.fit_transform(train_documents)
test_X = vectorizer.transform(test_documents)
print(train_X.shape, test_X.shape)
```

测试文档若只有训练词表外的词，其向量可能是全零行；词面模型无法从未见 token 推断语义。这是 OOV（out-of-vocabulary）策略的结果，不表示新词已被加入训练特征。

## 17. scikit-learn 的 TfidfVectorizer

TfidfVectorizer 将词表学习、TF-IDF 计算和归一化封装起来，输出通常是稀疏矩阵。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "machine learning is useful",
    "deep learning is powerful",
    "natural language processing",
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.shape)
print(X.toarray().round(3))
```

对每一行做 L2 归一化后，常用点积就等于余弦相似度。TfidfVectorizer 默认通常会进行归一化，但应查看当前版本配置，不要只凭记忆。

```python
print(np.linalg.norm(X.toarray(), axis=1))
```

重要参数包括：

- ngram_range：使用哪些 N-gram；
- min_df/max_df：过滤过稀或过常见特征；
- max_features：限制最大特征数；
- stop_words：停用词；
- sublinear_tf：对 TF 做对数缩放；
- analyzer/tokenizer：词级或字符级特征。

```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_features=1000,
    sublinear_tf=True,
)
X = vectorizer.fit_transform(documents)
print(X.shape)
```

## 18. 中文 TF-IDF：jieba 加 TfidfVectorizer

中文文本没有空格，通常先用 jieba 分词，再将 token 传给 TfidfVectorizer。自定义 tokenizer 时要关闭默认 token_pattern，避免 scikit-learn 二次按英文规则切分。

```python
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "我喜欢自然语言处理",
    "自然语言处理很有趣",
    "机器学习可以分析文本",
]

def chinese_tokenizer(text):
    return [word for word in jieba.lcut(text) if word.strip()]

vectorizer = TfidfVectorizer(
    tokenizer=chinese_tokenizer,
    token_pattern=None,
)
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(X.toarray().round(3))
```

可在 tokenizer 中过滤标点或停用词，但过滤规则应只根据训练任务确定，并在推理中完全复用。

```python
stop_words = {"的", "了", "和", "是", "在"}

def clean_chinese_tokenizer(text):
    tokens = jieba.lcut(text)
    return [
        token for token in tokens
        if token.strip() and token not in stop_words
    ]

vectorizer = TfidfVectorizer(
    tokenizer=clean_chinese_tokenizer,
    token_pattern=None,
)
print(vectorizer.fit_transform(documents).shape)
```

## 19. TF-IDF 与关键词提取

对于一篇文档，TF-IDF 高的词常是该文档相对突出的词。可以取某一行向量的 top-k 维度作为关键词候选。

```python
documents = [
    "人工智能推动机器学习发展",
    "机器学习可以分析数据",
    "自然语言处理分析文本",
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
terms = vectorizer.get_feature_names_out()

row = X[0].toarray().ravel()
top_indices = row.argsort()[::-1][:3]
print([(terms[i], row[i]) for i in top_indices])
```

TF-IDF 关键词只是统计意义上的重要性，不等于人工定义的主题，也不一定适合短文本。停用词、词粒度和语料规模都会改变结果。

```python
for document_index in range(X.shape[0]):
    row = X[document_index].toarray().ravel()
    top = row.argsort()[::-1][:2]
    print(document_index, [terms[i] for i in top])
```

## 20. 文本相似度与余弦距离

把文档表示成向量后，可以使用余弦相似度比较方向。它不直接关注向量长度，适合比较词频分布相似性。

```python
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "machine learning models",
    "machine learning algorithms",
    "cooking recipes",
]
X = TfidfVectorizer().fit_transform(documents)
similarity = cosine_similarity(X)
print(similarity.round(3))
```

余弦相似度范围通常在 -1 到 1；TF-IDF 非负时常见结果在 0 到 1。相似度高表示词汇分布相近，不保证语义真正相同。

```python
query = TfidfVectorizer().fit(documents)
X = query.transform(documents)
query_vector = query.transform(["learning model"])
scores = cosine_similarity(query_vector, X).ravel()
print(scores)
print(scores.argsort()[::-1])
```

查询必须使用与文档相同的 vectorizer。不能为 query 重新 fit 一个词表，否则各列含义不一致。

## 21. 文档检索的最小流程

传统检索流程是：对文档集合 fit 向量器，对文档 transform，查询使用同一向量器 transform，再计算相似度并排序。

```python
documents = [
    "PyTorch supports tensor computation",
    "TF IDF represents documents with statistics",
    "Chinese segmentation uses jieba",
]
vectorizer = TfidfVectorizer()
document_matrix = vectorizer.fit_transform(documents)

query_vector = vectorizer.transform(["tensor computation"])
scores = cosine_similarity(query_vector, document_matrix).ravel()
ranking = scores.argsort()[::-1]

for index in ranking:
    print(scores[index], documents[index])
```

这种方法透明、部署简单，适合小型搜索和 baseline；它对词面变化、同义词和上下文语义不敏感。

```python
queries = ["tensor", "统计文本", "分词"]
for query_text in queries:
    query_vector = vectorizer.transform([query_text])
    scores = cosine_similarity(query_vector, document_matrix).ravel()
    print(query_text, scores.round(3))
```

查询完全没有训练词表中的词时，可能得到全零向量，所有相似度都接近 0。生产系统应处理这种 OOV 情形。

```python
query_vector = vectorizer.transform(["completely_unknown_term"])
print(query_vector.nnz)
```

## 22. 传统表示与分类模型

BoW/TF-IDF 常与 Logistic Regression、Linear SVM、朴素贝叶斯等线性或浅层模型组合。稀疏文本向量通常适合线性分类器，因为模型可以直接学习每个词或 N-gram 的权重。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

texts = [
    "这部电影非常精彩",
    "这部电影非常糟糕",
    "故事紧凑值得推荐",
    "故事拖沓不值得观看",
]
labels = [1, 0, 1, 0]

model = make_pipeline(
    TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None),
    LogisticRegression(max_iter=1000),
)
model.fit(texts, labels)
print(model.predict(["这部电影值得推荐"]))
```

Pipeline 将向量器和分类器绑定，调用 fit 时只在训练数据上学习词表和 IDF，预测时复用相同变换。

```python
print(model.predict_proba(["剧情很好，值得观看"]))
```

在真实数据上应使用独立验证集、交叉验证和分类别指标，不能因为小例子预测正确就认为模型泛化良好。

## 23. 数据切分与数据泄漏

向量器的 fit 会学习词表、IDF、停用词统计或特征选择结果，因此必须只在训练集执行。推荐把向量器和分类器放入 Pipeline，并把 Pipeline 交给交叉验证。

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

train_x, test_x, train_y, test_y = train_test_split(
    texts,
    labels,
    test_size=0.5,
    random_state=42,
    stratify=labels,
)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        token_pattern=None,
    )),
    ("classifier", LogisticRegression(max_iter=1000)),
])
pipeline.fit(train_x, train_y)
print(pipeline.score(test_x, test_y))
```

错误方式是先对所有文本 fit_transform，再拆训练/测试。即使只是词表出现情况，也会把测试分布提前暴露给训练流程。

```python
# 不推荐：全量文本参与 fit
all_vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None)
all_X = all_vectorizer.fit_transform(texts)
train_X = all_X[:4]
test_X = all_X[4:]
print(train_X.shape, test_X.shape)
```

## 24. 特征选择与维度控制

N-gram 和大语料会产生大量特征。可以用 min_df、max_df、max_features，或使用 SelectKBest 等方法控制维度。特征选择同样必须只在训练集拟合。

```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=50000,
)
print(vectorizer)
```

min_df 可以是整数或比例；太大可能删除重要的领域词，太小可能保留大量拼写噪声。应结合特征数量、验证指标和推理内存选择。

```python
vectorizer = TfidfVectorizer(max_features=3)
X = vectorizer.fit_transform([
    "cat dog fish",
    "cat dog",
    "bird fish",
])
print(vectorizer.get_feature_names_out())
```

## 25. 稀疏矩阵与内存

传统词袋和 TF-IDF 矩阵通常绝大多数位置为 0，scikit-learn 默认返回 scipy sparse matrix。盲目调用 toarray 会把它转成 dense 矩阵，可能造成内存暴涨。

```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(["cat dog", "dog fish", "cat"])
print(type(X))
print("shape:", X.shape)
print("nonzero:", X.nnz)
```

密度可以用非零元素数除以总元素数估算：

```python
density = X.nnz / (X.shape[0] * X.shape[1])
print("density:", density)
```

只有在特征很少、需要打印或交给只支持 dense 的算法时才转 dense。生产中优先使用支持稀疏输入的模型。

```python
dense = X.toarray()
print(dense)
```

## 26. 哈希向量化

HashingVectorizer 不保存显式词表，而是使用哈希函数把 token 映射到固定数量的桶。优点是内存固定、适合流式数据；缺点是发生 hash collision，不能直接反查每个维度对应哪个词，也不能通过 fit 学习 IDF。

```python
from sklearn.feature_extraction.text import HashingVectorizer

hasher = HashingVectorizer(n_features=8, alternate_sign=False)
X = hasher.transform(["machine learning", "natural language"])
print(X.shape)
print(X.toarray())
```

n_features 越小，碰撞概率通常越高；越大则占用更多特征空间。HashingVectorizer 可以和在线分类器组合处理不断到来的文本。

```python
from sklearn.linear_model import SGDClassifier

hasher = HashingVectorizer(n_features=2**12, alternate_sign=False)
classifier = SGDClassifier(loss="log_loss", random_state=42)
X = hasher.transform(["good movie", "bad movie"])
classifier.partial_fit(X, [1, 0], classes=[0, 1])
print(classifier.predict(hasher.transform(["good film"])))
```

## 27. 共现矩阵

BoW 关注文档-词关系，共现矩阵可以统计词-词在窗口内共同出现的次数。它是分布式表示和传统语义分析的重要思想：出现在相似上下文中的词可能具有相似语义。

```python
from collections import defaultdict

tokens = "自然语言处理 需要 语言模型 语言 数据".split()
window = 2
cooccurrence = defaultdict(Counter)

for i, token in enumerate(tokens):
    for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
        if i != j:
            cooccurrence[token][tokens[j]] += 1

for token, counts in cooccurrence.items():
    print(token, counts)
```

共现矩阵可能很大、很稀疏，而且窗口大小、边界和词频都会影响统计结果。它描述上下文共现，不等同于经过训练的语义 embedding。

```python
vocabulary = sorted(set(tokens))
co_matrix = np.zeros((len(vocabulary), len(vocabulary)), dtype=np.float32)
index = {word: i for i, word in enumerate(vocabulary)}

for word, counts in cooccurrence.items():
    for context, count in counts.items():
        co_matrix[index[word], index[context]] = count

print(vocabulary)
print(co_matrix)
```

## 28. 从共现矩阵到低维表示

共现矩阵可以使用 SVD 分解成低维表示。LSA/LSI 用线性代数压缩高维词-文档或词-词矩阵，减少噪声并得到稠密向量。

```python
from sklearn.decomposition import TruncatedSVD

matrix = np.array([
    [3, 0, 1, 0],
    [2, 0, 0, 1],
    [0, 3, 0, 1],
    [0, 2, 0, 2],
], dtype=np.float32)

svd = TruncatedSVD(n_components=2, random_state=42)
low_dim = svd.fit_transform(matrix)
print(low_dim)
print(svd.explained_variance_ratio_)
```

SVD 得到的方向由当前语料决定，换语料、词表或预处理就可能变化。维度数量需要在验证任务上选择，不能把降到二维就理解为真实语义空间。

```python
similarity = cosine_similarity(low_dim)
print(similarity.round(3))
```

## 29. BoW、TF-IDF 与神经 Embedding 对比

| 表示 | 主要依据 | 是否稠密 | 是否学习语义 | 是否保留词序 |
| --- | --- | --- | --- | --- |
| One-Hot | ID 索引 | 否 | 否 | 否 |
| BoW | 文档内词频 | 否 | 否 | 否 |
| N-gram | 局部连续片段 | 否 | 否 | 保留有限局部顺序 |
| TF-IDF | TF 和 IDF | 否 | 否 | 取决于 N-gram |
| SVD/LSA | 共现/词文档矩阵分解 | 是 | 间接获得 | 间接 |
| nn.Embedding | 训练目标学习 | 是 | 是 | 由下游模型决定 |

```python
# One-Hot：维度等于词表
one_hot = F.one_hot(torch.tensor([0, 2]), num_classes=4).float()
# 神经 Embedding：输出维度由 embedding_dim 决定
embedding = nn.Embedding(4, 3)
dense = embedding(torch.tensor([0, 2]))
print(one_hot.shape, dense.shape)
```

TF-IDF 的“重要”是相对于当前文档集合的统计概念；神经 embedding 的“相似”取决于训练目标和数据。二者不能只看向量维度就互相替换。

## 30. 传统向量与 PyTorch 模型

scikit-learn 通常输出 scipy 稀疏矩阵，而 PyTorch MLP 常接收 dense Tensor。小数据可以转成 dense；大数据要注意内存，或使用支持稀疏输入的模型。

```python
from torch.utils.data import TensorDataset, DataLoader

documents = ["cat dog", "dog fish", "cat bird"]
vectorizer = TfidfVectorizer()
X_sparse = vectorizer.fit_transform(documents)
X_dense = torch.tensor(X_sparse.toarray(), dtype=torch.float32)
labels = torch.tensor([0, 1, 0], dtype=torch.long)

loader = DataLoader(TensorDataset(X_dense, labels), batch_size=2)
for features, target in loader:
    print(features.shape, target.shape)
```

可以用一个线性层处理 TF-IDF 特征：

```python
model = nn.Linear(X_dense.shape[1], 2)
for features, target in loader:
    logits = model(features)
    loss = F.cross_entropy(logits, target)
    print(logits.shape, loss.item())
```

这不是把 TF-IDF 变成神经 embedding；它是把统计特征作为模型输入，再让线性层学习分类决策边界。

## 31. 文档向量和词向量的区别

One-Hot 通常表示一个 token，BoW/TF-IDF 通常表示一篇文档。文档向量的维度是词表/特征表大小，词向量的维度可以是词表大小或压缩后的低维 embedding。不能把“某文档 TF-IDF 第 i 维”直接当作“词 i 的语义向量”。

```python
documents = ["cat dog", "dog fish"]
vectorizer = TfidfVectorizer()
document_matrix = vectorizer.fit_transform(documents)
print("document matrix:", document_matrix.shape)
print("terms:", vectorizer.get_feature_names_out())
```

如果要得到固定维的句向量，可以对 token embedding 做平均池化、使用专用 sentence embedding 模型，或对 TF-IDF 做降维；这些方法的语义性质不同。

```python
embedding = nn.Embedding(10, 4, padding_idx=0)
ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
vectors = embedding(ids)
mask = ids.ne(0).unsqueeze(-1)
sentence_vectors = (vectors * mask).sum(1) / mask.sum(1).clamp_min(1)
print(sentence_vectors.shape)
```

## 32. 特征值域和标准化

TF-IDF 通常非负，且常做 L2 归一化；线性模型的正则化和学习率会受特征尺度影响。使用不同向量化方式时，应检查数值范围。

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(["cat dog", "dog fish"])
print(X.min(), X.max())
print(np.linalg.norm(X.toarray(), axis=1))
```

将统计特征送入需要标准化的模型时，应只在训练集拟合 scaler，并将相同 scaler 应用于验证/测试。

```python
from sklearn.preprocessing import StandardScaler

X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
```

对稀疏矩阵不要随意使用会破坏稀疏性的中心化操作。应查 StandardScaler 的 sparse 支持和参数约定，或选用适合稀疏特征的模型。

```python
scaler = StandardScaler(with_mean=False)
print(scaler.fit_transform(X))
```

## 33. 常见错误

| 错误 | 原因 | 修正 |
| --- | --- | --- |
| 用 token ID 直接做连续特征 | ID 没有大小/距离语义 | One-Hot、BoW、TF-IDF 或 Embedding |
| 中文直接使用默认英文 tokenizer | 中文没有空格词边界 | jieba 或适配中文的 tokenizer |
| 训练集/测试集分别 fit | 列含义和 IDF 不一致 | 训练集 fit，其他集合 transform |
| 对全量数据先 fit | 测试分布泄漏 | 使用 Pipeline 和正确切分 |
| N-gram 特征太多 | 维度和稀疏性爆炸 | min_df、max_features、限制 n |
| 稀疏矩阵无条件 toarray | 内存暴涨 | 保持 sparse 或分块处理 |
| TF-IDF 相似度低就认为语义不相似 | 词面方法不理解同义词 | 使用语义 embedding 或混合检索 |
| 词频高就认为关键词重要 | 高频不等于区分度高 | TF-IDF、停用词和验证 |
| 修改词表后继续使用旧模型 | 特征列语义变化 | 重新训练或保存完整 vectorizer |

```python
def validate_vocabulary(vectorizer, matrix):
    names = vectorizer.get_feature_names_out()
    assert matrix.shape[1] == len(names)
    print("features:", matrix.shape[1])

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(["cat dog", "dog fish"])
validate_vocabulary(vectorizer, matrix)
```

## 34. 传统表示的适用场景

One-Hot 适合解释离散类别或理解 Embedding 的数学基础。BoW/TF-IDF + 线性模型适合快速 baseline、小中型数据、特征解释和检索。N-gram 适合需要局部词序或短语信息的任务。字符 N-gram 对拼写变化和未登录词有一定鲁棒性。

```python
choices = {
    "One-Hot": "教学、离散类别、小词表",
    "BoW": "快速词频基线、可解释分类",
    "N-gram": "保留局部短语和词序",
    "TF-IDF": "检索、关键词、传统分类",
    "神经 Embedding": "需要可学习的稠密表示和下游深度模型",
}
for method, use in choices.items():
    print(method, "->", use)
```

大规模语义检索、跨句上下文、同义表达和复杂生成通常需要预训练 embedding 或 Transformer。传统方法仍适合作为 baseline 和可解释的补充。

```python
baseline = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
print(baseline)
```

## 35. 完整中文传统表示流程

一个可靠的中文传统表示流程通常包括：文本清理、jieba 分词、停用词处理、训练集 fit TF-IDF、分类器训练、独立评估。每一步都要保存配置。

```python
import jieba
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenizer(text):
    return [token for token in jieba.lcut(text) if token.strip()]

texts = [
    "这部电影非常精彩",
    "演员表演很出色",
    "这部电影非常糟糕",
    "剧情拖沓令人失望",
    "故事温暖值得推荐",
    "故事混乱不值得观看",
]
labels = [1, 1, 0, 0, 1, 0]

pipeline = make_pipeline(
    TfidfVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,
        ngram_range=(1, 2),
    ),
    LogisticRegression(max_iter=1000),
)
pipeline.fit(texts, labels)
print(pipeline.predict(["剧情精彩，值得推荐"]))
```

真实项目还需要拆分验证集、检查类别平衡、查看误分类样本，并避免使用测试数据调节停用词和 N-gram 参数。

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, texts, labels, cv=3)
print(scores, scores.mean())
```

## 36. 与神经 Embedding 的衔接

传统表示可以帮助理解神经 Embedding 的动机：One-Hot 是高维稀疏身份表示，Embedding 是将每个 ID 映射到低维稠密空间。神经 Embedding 的向量不是由 One-Hot 本身提供语义，而是由训练目标调整参数。

```python
vocab_size, embedding_dim = 6, 3
one_hot = F.one_hot(torch.tensor([1, 4]), num_classes=vocab_size).float()
embedding = nn.Embedding(vocab_size, embedding_dim)
dense_vectors = embedding(torch.tensor([1, 4]))
print(one_hot.shape, dense_vectors.shape)
```

可以把 Embedding 看作 One-Hot 与可学习矩阵相乘的高效实现，但它仍需要下游训练目标才能形成有用表示。

```python
with torch.no_grad():
    weight = embedding.weight.clone()
manual = one_hot @ weight
print(torch.allclose(manual, dense_vectors))
```

传统 TF-IDF 与神经 Embedding 也可以组合：例如用 TF-IDF 召回候选，再用神经 embedding 重排。但两种表示的词表、维度和相似度空间不能混用。

```python
tfidf_query = TfidfVectorizer().fit_transform(["文本检索"])
neural_query = torch.randn(1, 16)
print(tfidf_query.shape, neural_query.shape)
```

## 37. 学习与实验路线

建议按以下顺序实践：

```python
steps = [
    "手写词表和 One-Hot",
    "手写 BoW 并处理未知词",
    "用 CountVectorizer 验证词频矩阵",
    "加入 unigram/bigram 比较局部顺序",
    "手算 TF、DF、IDF 和 TF-IDF",
    "用 TfidfVectorizer 做相似度检索",
    "用 jieba + TF-IDF + 线性分类器做中文 baseline",
    "比较 TF-IDF 与 nn.Embedding 分类器",
]
for number, step in enumerate(steps, 1):
    print(number, step)
```

每次实验至少记录 tokenizer、词表大小、N-gram 范围、停用词、min_df、max_features、模型、验证指标和错误案例。

```python
experiment_config = {
    "tokenizer": "whitespace or jieba",
    "representation": "tfidf",
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_features": None,
}
print(experiment_config)
```

## 38. 总结

传统 NLP 文本表示的主线是：

```text
字符串
  -> token
  -> 词表
  -> One-Hot / BoW / N-gram / TF-IDF
  -> 稀疏特征矩阵
  -> 相似度、检索、分类或降维
```

需要记住：

- One-Hot 表示 token 身份，维度高且不表达相似度；
- BoW 将 token 计数聚合成文档向量，但忽略词序；
- N-gram 保留有限局部顺序，却带来更多稀疏特征；
- TF-IDF 降低语料中普遍词的权重，突出文档区分词；
- 词表、IDF 和特征选择必须只在训练数据上 fit；
- scikit-learn 通常返回稀疏矩阵，不要无条件转 dense；
- jieba 适合传统中文词级特征，Transformer 应使用配套 tokenizer；
- 神经 Embedding 是可训练的低维稠密表示，语义来自训练目标而不是 ID；
- 传统 TF-IDF 适合 baseline、检索和可解释特征，神经 embedding 适合语义建模。

理解这些传统表示后，再学习 nn.Embedding、Word2Vec、Transformer 和句向量，会更容易理解“文本如何从离散符号变成模型可以计算的向量”。
