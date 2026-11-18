# NLTK 自然语言处理工具包：从文本基础到传统 NLP

NLTK（Natural Language Toolkit）是 Python 中经典的自然语言处理教学与研究工具包。它提供文本切分、词形处理、词性标注、语料库、词频统计、n-gram、朴素贝叶斯分类等功能。学习 NLTK 的重点不是把它当作现代大语言模型框架，而是借助清晰、可拆解的工具理解 NLP 的基本处理流程。

本文从安装和资源管理开始，逐步讲解文本预处理、分词、停用词、词干提取、词形还原、词性标注、命名实体识别、语料统计、特征构造与传统分类。每个知识点都配有示例代码。NLTK 示例以英文为主；NLTK 的英文资源和模型不能直接当作中文分词器使用。中文部分会说明适用边界并给出 `jieba` 示例。

> NLTK 的不少功能依赖单独下载的语料或模型。Python 包安装成功，不代表所需资源已经下载。示例会明确列出资源名；资源下载通常需要网络，建议在独立环境中完成并记录资源版本。

## 1. NLTK 能做什么

处理文本时，原始字符串通常要经过多个步骤才能变成可统计、可训练的数据。NLTK 主要覆盖经典 NLP 管线中的基础处理和教学功能：切分句子/单词、清理 token、词形归一化、查词典或语料、抽取局部特征、训练简单分类器。

```text
原始文本
  -> 句子切分
  -> 单词/标点切分
  -> 规范化与过滤
  -> 词形处理、词性或实体标注
  -> 统计与特征
  -> 传统机器学习模型
```

```python
import nltk

text = "NLTK helps us explore language with Python."
sentences = nltk.sent_tokenize(text, language="english")
tokens = nltk.word_tokenize(text, language="english")
print(sentences)
print(tokens)
```

NLTK 是通用 NLP 教学/研究工具集，不等同于神经网络模型，也不自动完成高质量文本理解。实际项目应把数据规范、tokenizer、词表、标签、模型和评估作为一个整体设计。

## 2. 安装和环境检查

安装 NLTK Python 包只安装了代码；语料库和预训练处理器通常另行下载。使用项目 `uv` 环境时可从项目根目录安装依赖，也可在普通 Python 环境使用 pip。

```bash
# 项目已在 pyproject.toml 声明 nltk 时，同步项目依赖
uv sync

# 独立环境的替代方式
python -m pip install nltk
```

安装后检查 Python 与 NLTK 版本，确保命令执行的解释器就是运行 Notebook/脚本的解释器。

```python
import sys
import nltk

print("Python:", sys.executable)
print("NLTK:", nltk.__version__)
```

## 3. NLTK 数据资源与下载

NLTK 将大型语料和模型作为独立数据资源提供。Tokenizer、POS tagger、WordNet lemmatizer 等组件可能分别依赖不同资源。`LookupError` 通常表示包已安装，但对应数据没有安装或 NLTK 没有在配置目录中找到它。

```python
import nltk

# 只下载本学习文档用到的资源，避免不必要地下载整套语料
for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4", "stopwords", "movie_reviews"]:
    nltk.download(resource, quiet=False)
```

不同 NLTK 版本使用的资源名称和组件要求可能变化。例如新版句子/单词切分可能需要 `punkt_tab`，而英文 POS tagger 可能需要 `averaged_perceptron_tagger_eng`。若仍报 `LookupError`，以异常中提示的路径/资源名和当前 NLTK 版本为准，不要假定只下载 `punkt` 就能解决所有问题。

```python
import nltk

print(nltk.data.path)  # NLTK 搜索数据资源的目录
try:
    nltk.data.find("tokenizers/punkt_tab/english/")
    print("punkt_tab 已找到")
except LookupError:
    print("请下载 punkt_tab，或确认 NLTK_DATA 指向正确目录")
```

下载资源到自定义目录时，把该目录加入搜索路径。生产/离线环境可以预先下载经过审核的资源并随环境部署；不要在每次程序启动时依赖公网下载。

```python
from pathlib import Path
import nltk

data_dir = Path("./nltk_data").resolve()
data_dir.mkdir(parents=True, exist_ok=True)
nltk.download("punkt_tab", download_dir=str(data_dir))
nltk.data.path.append(str(data_dir))
```

## 4. 文本规范化：先明确规则，再动手清理

规范化（normalization）是把不同表面形式变为相对一致的形式，例如去除首尾空格、统一大小写、处理 Unicode 或压缩空白。清理不等于越多越好：大小写可能包含专名信息，标点可能表示语气或句界，数字可能是分类任务的关键信息。应根据任务定义保留或删除。

```python
import re
import unicodedata

text = "  Café   NLP!  "
text_nfc = unicodedata.normalize("NFC", text)
text_clean = re.sub(r"\s+", " ", text_nfc).strip()
text_lower = text_clean.lower()
print(text_nfc, text_clean, text_lower)
```

Unicode 中一个视觉字符可能由不同码点序列构成。NFC 常把可组合字符规范为一致形式；NFKC 更激进，会把部分兼容字符折叠。不要在不了解数据语义时对所有文本无条件使用 NFKC。

```python
import unicodedata

composed = "é"
decomposed = "e\u0301"
print(composed == decomposed)  # False：字符串编码形式不同
print(unicodedata.normalize("NFC", composed) == unicodedata.normalize("NFC", decomposed))
```

## 5. 句子切分

Sentence tokenization 把一段文本划分为句子。简单地按句号切分会误处理缩写（如 `Dr.`）、小数、引号和省略号；NLTK 的 Punkt 方法尝试从句界线索识别句子边界，但仍受语言、领域和文本格式影响。

```python
import nltk

paragraph = "Dr. Smith studies NLP. The result is useful! Is it ready?"
sentences = nltk.sent_tokenize(paragraph, language="english")
for sentence in sentences:
    print(repr(sentence))
```

如果是英文以外的语言，应指定对应语言并确认所需资源已安装。句子切分是预处理选择，不保证符合每个领域的标注规范；法律文本、聊天记录、代码混合文本常需要专门规则。

```python
text = "First sentence. Second sentence!"
sentences = nltk.sent_tokenize(text, language="english")
print(len(sentences), sentences)
```

## 6. 单词切分与标点

Word tokenization 把句子拆成 token。NLTK 的 `word_tokenize` 通常会将标点拆成独立 token，并使用句子切分作为处理的一部分。token 不一定等于语言学意义上的“词”：缩写、所有格、连字符词和 emoji 都有 tokenizer 约定。

```python
from nltk.tokenize import word_tokenize

text = "NLTK's tokenizer handles words, punctuation, and can't be reduced to split()."
tokens = word_tokenize(text, language="english")
print(tokens)
```

按空格 `split()` 是最简单的切分方式，但会把附着在词上的标点留在词内，也不能处理多空格、换行和部分语言规则。正则 tokenizer 可用于自定义简单规则，但不是通用语言分析器。

```python
import re

text = "Hello, world! Version 3.10"
whitespace_tokens = text.split()
regex_tokens = re.findall(r"\w+|[^\w\s]", text)
print(whitespace_tokens)
print(regex_tokens)
```

NLTK 提供多种 tokenization 接口。`TreebankWordTokenizer` 可在无需句子模型下载时演示英语规则，但它与 `word_tokenize` 的细节不完全相同，结果应按任务验证。

```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
print(tokenizer.tokenize("They'll save time, won't they?"))
```

## 7. 中文文本切分的边界

NLTK 的 `word_tokenize` 面向英语等基于空格的语言，不会自动完成可靠的中文词语边界识别。对中文可用专用分词器（如 jieba）或直接使用适配中文的子词 tokenizer；具体方案取决于传统特征工程还是预训练模型输入。

```python
# pip install jieba
import jieba

text = "自然语言处理需要结合上下文理解词语。"
tokens = list(jieba.cut(text))
print(tokens)
```

中文分词存在歧义，例如“研究生命起源”可以出现不同切分。分词器版本、词典和用户词典都会影响结果。使用中文 Transformer 时一般加载其配套 tokenizer，不应先用 NLTK 或 jieba 随意切分后再传入模型。

```python
sentence = "南京市长江大桥"
print(list(jieba.cut(sentence, cut_all=False)))
print(list(jieba.cut(sentence, cut_all=True)))  # 全模式可能给出更多候选片段
```

## 8. 统一大小写与过滤标点

英文传统文本分类常把 token 变成小写，减少词表中 `Book`/`book` 等重复项；但大小写也可能区分人名、组织名、缩写或句首信息。标点过滤同样需要按任务决定。处理代码应显式表达规则，而不是在不了解数据时套一串清理操作。

```python
tokens = ["NLP", "is", "Useful", "!", "Really", "?"]
lower_tokens = [token.lower() for token in tokens]
word_only = [token for token in lower_tokens if any(char.isalpha() for char in token)]
print(lower_tokens)
print(word_only)
```

更谨慎的规则可保留字母/数字 token，并只移除纯标点；不要误删 `C++`、`COVID-19`、`U.S.` 等可能有意义的写法。

```python
import string

tokens = ["nlp", "C++", "covid-19", ",", "42"]
without_punctuation = [token for token in tokens if token not in string.punctuation]
print(without_punctuation)
```

## 9. 停用词：定义、用途与风险

停用词（stop words）是任务中常被认为信息较少的高频词，如英文 `the`、`is`。移除它们可以缩小词袋特征空间，但也会改变句意：否定词 `not`、`no` 对情感和事实判断非常关键。NLTK 的停用词表是通用资源，不是适用于所有任务的权威名单。

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
tokens = word_tokenize("This movie is not good, but the music is beautiful.")
filtered = [token for token in tokens if token.lower() not in stop_words]
print(filtered)
```

对情感任务，应检查停用词表是否包含否定词，并根据验证集效果决定是否保留。中文停用词表通常需从项目指定来源获取、检查许可和编码，再结合任务维护。

```python
stop_words = set(stopwords.words("english"))
for important_word in ["not", "no", "never"]:
    stop_words.discard(important_word)
print([word for word in ["not", "very", "good"] if word.lower() not in stop_words])
```

## 10. 词干提取 Stemming

词干提取器通过规则去掉词尾，试图把词形变体归并到相同字符串。结果可能不是合法单词，但速度快，不依赖词典。Porter stemmer 是英语经典算法；它的目标是规则化，不是理解语义。

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["connect", "connected", "connecting", "connection"]
print([(word, stemmer.stem(word)) for word in words])
```

词干会因算法而异，且可能过度截短或把相关形式错误合并。Snowball stemmer 提供多个语言的规则实现；不能把英语 Porter 规则直接用于中文。

```python
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
print(stemmer.stem("generously"))
```

## 11. 词形还原 Lemmatization

词形还原把词还原到词典中的标准形式（lemma），通常比 stemming 更依赖词典或词性信息。WordNetLemmatizer 默认按名词方式处理；动词、形容词等应提供正确 POS，否则结果可能看起来没有变化。

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
for word in ["cars", "running", "better"]:
    print(word, "->", lemmatizer.lemmatize(word))
```

POS 参数使用 WordNet 的标签，例如 `n` 名词、`v` 动词、`a` 形容词、`r` 副词。下面对同一个表面形式分别指定名词/动词，展示词性会改变还原结果。

```python
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("saw", pos="n"))
print(lemmatizer.lemmatize("saw", pos="v"))
```

若文本是完整句子，可先做 POS tagging，再把 Penn Treebank 标签映射到 WordNet 标签。词性标注需要相应模型资源；标注错误也会传递到 lemma。

```python
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

def penn_to_wordnet(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    if tag.startswith("N"):
        return wordnet.NOUN
    return wordnet.NOUN

tokens = word_tokenize("The children were running quickly.")
tagged = pos_tag(tokens, lang="eng")
lemmas = [lemmatizer.lemmatize(word, penn_to_wordnet(tag)) for word, tag in tagged]
print(tagged)
print(lemmas)
```

## 12. 词性标注 POS Tagging

词性标注为每个 token 预测名词、动词、形容词等语法类别。它利用词的形式和上下文，不能仅凭单词表查出唯一答案（如 `book` 可作名词或动词）。NLTK 的 `pos_tag` 使用预训练标注器，不等于手写规则。

```python
import nltk
from nltk.tokenize import word_tokenize

tokens = word_tokenize("The quick brown fox jumps over the lazy dog.")
tagged = nltk.pos_tag(tokens, lang="eng")
print(tagged)
```

标签常使用 Penn Treebank 体系，如 `NN` 名词、`NNS` 复数名词、`VB` 动词原形、`VBG` 现在分词、`JJ` 形容词。标签集是约定，不同语料和工具可能不同。

```python
tagged = [("cats", "NNS"), ("run", "VBP"), ("quickly", "RB")]
for word, tag in tagged:
    print(f"{word:10s} {tag}")
```

## 13. 命名实体识别 NER

命名实体识别试图识别人名、组织、地点等文本片段。NLTK 教学流程通常先对 token 做 POS 标注，再通过 `ne_chunk` 生成树结构。它依赖英文资源和特定标签体系，对领域文本、中文或现代社交媒体未必适用。

```python
import nltk
from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize

sentence = "Ada Lovelace worked in London."
tokens = word_tokenize(sentence)
tagged = pos_tag(tokens, lang="eng")
tree = ne_chunk(tagged, binary=False)
print(tree)
```

实体树中普通 token 是词/标签对，实体片段是带实体类别的子树。可递归提取实体；输出结果应结合上下文人工检查，不能把一次预测直接当作事实。

```python
def extract_entities(tree):
    entities = []
    for node in tree:
        if isinstance(node, nltk.Tree):
            name = " ".join(word for word, _tag in node.leaves())
            entities.append((name, node.label()))
    return entities

print(extract_entities(tree))
```

## 14. 词频统计与分布观察

在训练模型之前，先观察 token 频率有助于发现清理问题、语料偏差、异常符号和类别差异。频率高不代表信息量大，稀有词也不一定无用。NLTK 的 `FreqDist` 是基于计数的分布工具。

```python
from nltk import FreqDist
from nltk.tokenize import word_tokenize

text = "NLP studies text. Text contains words, and words have context."
tokens = [token.lower() for token in word_tokenize(text) if token.isalpha()]
freq = FreqDist(tokens)
print(freq.most_common(5))
print("unique tokens:", freq.B())
```

可以用频率分布查看特定词的计数或绘图。绘图需要图形环境，脚本/服务器中可以改为输出表格或保存图片。

```python
from nltk import FreqDist

freq = FreqDist(["cat", "dog", "cat", "bird", "dog", "cat"])
print(freq["cat"])
print(freq.freq("cat"))
# Notebook 中可执行：freq.plot(10, cumulative=False)
```

## 15. NLTK 内置语料库

NLTK 提供多种教学语料，可用于探索文本、词汇和标注。语料下载与软件包分离，使用前要下载相应集合。示例中的 `gutenberg` 是书籍文本集合；不同作品、语料具有各自许可和统计特性。

```python
import nltk
from nltk.corpus import gutenberg

nltk.download("gutenberg")
print(gutenberg.fileids()[:5])
words = gutenberg.words("austen-emma.txt")
print(len(words), words[:20])
```

语料库通常能按文件、词、句子或带标签样本读取。比较文档前应统一清理策略，并避免把语料库样例误当作生产任务的代表性数据。

```python
from nltk.corpus import gutenberg

file_id = "austen-emma.txt"
print("句子数:", len(gutenberg.sents(file_id)))
print("词数:", len(gutenberg.words(file_id)))
print("原始文本片段:", gutenberg.raw(file_id)[:200])
```

## 16. Concordance 与上下文探索

Concordance 按目标词列出其在语料中的局部上下文，便于观察多义、搭配和数据清理效果。它是探索工具，不是自动语义分析器；大小写、标点和分词会影响匹配结果。

```python
from nltk.book import text1

text1.concordance("monstrous", lines=5)
```

`nltk.book` 会加载教学语料并可能输出欢迎信息，适合 Notebook 探索。更可控的方式是自行创建 `Text` 并对本地文本做 tokenize。

```python
from nltk.text import Text
from nltk.tokenize import word_tokenize

tokens = word_tokenize("Language depends on context. Context changes meaning.")
corpus_text = Text(tokens)
corpus_text.concordance("Context", lines=3)
```

## 17. 搭配与共现

两个词频繁共同出现可能形成固定搭配，如 `New York`。只按原始共现次数会偏爱高频词；搭配指标会把实际共现与独立出现的期望进行比较。小语料上的高分可能偶然产生，需检查样本量和上下文。

```python
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
from nltk.tokenize import word_tokenize

tokens = word_tokenize("New York is busy. New York has many people. New York is large.")
finder = BigramCollocationFinder.from_words([token.lower() for token in tokens])
finder.apply_freq_filter(2)
print(finder.nbest(BigramAssocMeasures.pmi, 10))
```

PMI 对低频共现较敏感，因此常通过最低频次过滤控制噪声。搭配分析可用于特征探索、术语发现，但不能直接证明词语之间存在因果或稳定语义关系。

```python
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

finder = BigramCollocationFinder.from_words("data science data science machine learning".split())
finder.apply_freq_filter(2)
print(finder.score_ngrams(BigramAssocMeasures.raw_freq))
```

## 18. N-gram 与局部语言结构

N-gram 是连续 `n` 个 token 组成的序列。Unigram 统计单词，bigram 统计相邻词对，trigram 统计三词组合。N-gram 能捕获有限局部顺序，但组合数随词表和 n 增长，未见过的序列需要平滑或回退策略。

```python
from nltk.util import ngrams

tokens = ["natural", "language", "processing", "is", "useful"]
print(list(ngrams(tokens, 1)))
print(list(ngrams(tokens, 2)))
print(list(ngrams(tokens, 3)))
```

语言模型中的 n-gram 近似假设下一个词只依赖有限长度历史。例如 bigram 模型近似 `P(w_t | w_1...w_{t-1})` 为 `P(w_t | w_{t-1})`。这种马尔可夫假设简单但无法直接利用远距离上下文。

```python
from collections import Counter
from nltk.util import ngrams

tokens = "we learn language and we learn models".split()
bigram_counts = Counter(ngrams(tokens, 2))
context_counts = Counter(tokens[:-1])
context, next_word = "we", "learn"
probability = bigram_counts[(context, next_word)] / context_counts[context]
print(probability)  # 未平滑的最大似然估计
```

未平滑的最大似然估计会让任何未见过的 bigram 概率等于零。真实语言建模需要平滑、回退或神经语言模型；本示例仅说明计数概率如何得到。

```python
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

train_data = ["we learn language".split(), "we build models".split()]
train_ngrams, vocab = padded_everygram_pipeline(2, train_data)
lm = Laplace(2)
lm.fit(train_ngrams, vocab)
print(lm.score("learn", ["we"]))
```

## 19. 词袋特征与词汇表

词袋把文档映射为词项计数，不保留原始顺序。NLTK `FreqDist` 可帮助理解计数；生产分类通常需要训练集拟合固定词汇表，并将文档向量化。词汇表只应由训练数据建立，测试数据不能参与 `fit`。

```python
from nltk import FreqDist
from nltk.tokenize import word_tokenize

documents = ["cats like milk", "dogs like walks"]
tokenized = [[word.lower() for word in word_tokenize(doc) if word.isalpha()] for doc in documents]
vocabulary = sorted({word for doc in tokenized for word in doc})
print(vocabulary)
```

对固定词汇表，可将每篇文档计数为一个向量。下面是最小教学实现；稀疏矩阵、TF-IDF 和更完整的词表管理通常使用 scikit-learn。

```python
from collections import Counter

vocabulary = ["cats", "dogs", "like", "milk", "walks"]
index = {word: i for i, word in enumerate(vocabulary)}
tokens = ["cats", "like", "milk"]
vector = [0] * len(vocabulary)
for token, count in Counter(tokens).items():
    if token in index:
        vector[index[token]] = count
print(vector)
```

## 20. 情感分类：NLTK 朴素贝叶斯

朴素贝叶斯根据 Bayes 定理估计类别概率，并假设给定类别后特征条件独立。这个假设对自然语言显然过度简化，但模型简单、训练快，适合作为传统基线和概率推理教学。

```python
from nltk.classify import NaiveBayesClassifier

train_examples = [
    ({"contains(great)": True, "contains(boring)": False}, "positive"),
    ({"contains(great)": False, "contains(boring)": True}, "negative"),
    ({"contains(great)": True, "contains(boring)": False}, "positive"),
    ({"contains(great)": False, "contains(boring)": True}, "negative"),
]
classifier = NaiveBayesClassifier.train(train_examples)
print(classifier.classify({"contains(great)": True, "contains(boring)": False}))
```

NLTK movie_reviews 语料包含正负面影评，可用来构造词存在与否特征。为避免随机性，应固定切分种子；同一词表/特征提取过程应一致应用于训练和测试。

```python
import random
import nltk
from nltk.corpus import movie_reviews

nltk.download("movie_reviews")
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
random.Random(42).shuffle(documents)
```

为教学方便，可选训练集高频词作为特征。`word_features` 需要只从训练数据建立；下面的完整流程将先切分，再从训练部分估计特征词表。

```python
from nltk import FreqDist
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

split = int(len(documents) * 0.8)
train_docs, test_docs = documents[:split], documents[split:]
train_freq = FreqDist(word.lower() for words, _label in train_docs for word in words if word.isalpha())
word_features = [word for word, _count in train_freq.most_common(2000)]

def document_features(words):
    present = {word.lower() for word in words}
    return {f"contains({word})": word in present for word in word_features}

train_set = [(document_features(words), label) for words, label in train_docs]
test_set = [(document_features(words), label) for words, label in test_docs]
classifier = NaiveBayesClassifier.train(train_set)
print("accuracy:", accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
```

这个教学基线刻意使用词存在特征，忽略顺序、否定范围、语境和词形变化。评估时还应查看混淆矩阵、各类 precision/recall 和错误样本，不能只报告一个准确率。

## 21. NLTK 数据结构：Token、Text、Tree

NLTK 不只有字符串函数，也定义了便于语料探索的数据结构。`Text` 封装 token 序列，提供 concordance 和词汇分布等方法；`Tree` 用于表示句法或实体树。了解返回类型可以避免把结构化结果误当成普通字符串列表。

```python
from nltk.text import Text

tokens = ["NLP", "is", "useful", ".", "NLP", "is", "broad", "."]
text = Text(tokens)
print(text.vocab().most_common())
print(text.count("NLP"))
```

POS 标注返回 `(token, tag)` 对，实体 chunking 返回嵌套树。先打印一个小结果，再写转换代码，可以确认真实 API 结构。

```python
from nltk import Tree

example = Tree("S", [
    ("Ada", "NNP"),
    Tree("PERSON", [("Lovelace", "NNP")]),
    ("worked", "VBD"),
])
print(example)
print(example.leaves())
```

## 22. 把 NLTK 作为预处理组件使用

NLTK 函数可组合成小型处理管线：句子切分、tokenization、lowercase、停用词过滤和词形还原。每一步都要保留清楚的输入/输出约定；若后续要用字符 offset 或还原原文，不能随意丢失 token 位置和标点。

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_english(text):
    tokens = nltk.word_tokenize(text, language="english")
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return [lemmatizer.lemmatize(token) for token in tokens]

print(preprocess_english("The cats were chasing mice in the gardens."))
```

上述管线使用名词默认 lemma，未用词性标注，因而不是最精确的形态分析；它也可能移除对任务重要的词。应在具体数据上比较原文、分词结果和处理结果，并用验证集评估预处理是否有帮助。

```python
sample = "The cats were chasing mice in the gardens."
print("raw:", sample)
print("processed:", preprocess_english(sample))
```

## 23. 常见错误和排查

NLTK 入门最常见的故障是缺少语料资源、语言参数不合适、把 token 与句子混淆、忘记在训练/测试间复用同一词汇表，以及误以为 NLTK 会自动处理中文。

```python
import nltk

try:
    print(nltk.word_tokenize("A short sentence.", language="english"))
except LookupError as error:
    print("NLTK resource missing; inspect resource name and data path:")
    print(error)
```

| 表现 | 原因方向 | 检查方式 |
| --- | --- | --- |
| `LookupError: Resource ... not found` | 所需语料/模型未下载，或搜索路径不含数据目录 | 根据错误资源名下载；查看 `nltk.data.path` |
| 句子切分结果不符合预期 | 缩写、数字、领域标点或语言不匹配 | 打印原始上下文；验证分句边界；调整规则/模型 |
| `word_tokenize` 不会切中文词 | API 不是中文分词器 | 使用适配中文的 tokenizer，如 jieba 或模型配套 tokenizer |
| lemma 没有变化 | 默认名词 POS，或输入已是 lemma | 指定词性；检查 POS tag 和 WordNet 资源 |
| 分类器测试准确率异常高 | 数据泄漏、重复样本、测试集参与词表构建 | 核对切分先后、重复和 feature vocabulary 来源 |
| NLTK 代码与旧教程不同 | 包版本或数据资源版本变化 | 检查当前版本和官方 API/错误提示 |

## 24. NLTK 与现代 NLP 工具的边界

NLTK 擅长教学、语料探索和传统 NLP 组件；scikit-learn 常用于 TF-IDF 与传统分类；spaCy 提供面向生产的高效 pipeline 和多语言模型；Hugging Face Transformers 提供预训练 Transformer 及其配套 tokenizer；PyTorch 用于构建和训练神经网络。它们可以组合使用，但 tokenization 约定不能随意混用。

```python
tool_roles = {
    "NLTK": "经典 NLP 教学、语料探索、规则和传统算法",
    "scikit-learn": "TF-IDF、传统分类器、数据切分与指标",
    "spaCy": "高效 NLP pipeline、词性/实体等任务模型",
    "Transformers": "预训练 Transformer、配套 tokenizer 与微调接口",
    "PyTorch": "张量计算、神经网络定义与训练",
}
for tool, role in tool_roles.items():
    print(tool, "->", role)
```

如果使用预训练模型，必须采用对应 tokenizer 与模型权重匹配的输入约定，不要把 NLTK token 列表直接假设为模型 token IDs。传统词袋模型和 Transformer 模型面对的是不同特征表示与训练范式。

```python
# 预训练模型应使用其配套 tokenizer，而不是 NLTK token ID
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("model-name")
# encoded = tokenizer("Natural language processing", return_tensors="pt")
# print(encoded["input_ids"])
```

## 25. 推荐练习路线

按照“观察文本 -> 处理文本 -> 统计 -> 建特征 -> 训练评估”的顺序练习。每一步先打印中间结果，确认理解了数据变化，再进入下一步；不要只复制整段分类器代码看最终 accuracy。

```python
exercises = [
    "比较 split、TreebankWordTokenizer 和 word_tokenize 对缩写/标点的处理",
    "比较 stemming 与 lemmatization，并实验不同 POS",
    "检查停用词表是否包含任务中的否定词",
    "使用 FreqDist 观察影评语料的词频与文档差异",
    "训练 movie_reviews 朴素贝叶斯基线并阅读误分类样本",
    "将英文例子替换为中文数据，明确选择中文分词与停用词方案",
]
for number, exercise in enumerate(exercises, 1):
    print(f"{number}. {exercise}")
```

完成后可以进一步比较 NLTK 词袋基线与 scikit-learn TF-IDF、PyTorch Embedding 分类器或预训练 Transformer。比较时使用同一个可靠切分和相同评估指标，并记录预处理、tokenizer 与训练数据，才能解释性能差异。

## 26. 总结

NLTK 的核心学习价值是让 NLP 处理过程变得可观察：字符串如何切成句子和 token，词形如何归一化，语料如何统计，局部共现如何构造特征，简单分类器如何从文本中学习。与此同时，NLTK 的规则、词表和英语资源有明确适用范围；中文处理和现代预训练模型应使用更匹配的工具与 tokenizer。

```text
NLTK 入门主线
安装包与资源
  -> 规范化与分句/分词
  -> 停用词、stemming、lemmatization
  -> POS、NER、语料统计与 n-gram
  -> 文本特征与传统分类
  -> 独立评估、错误分析与工具选型
```

学习 NLTK 后，建议继续深入词袋/TF-IDF、Tokenizer、Embedding、RNN/Attention 以及预训练模型微调。NLTK 给出的经典概念依旧有用，但具体项目要根据语言、领域、数据规模、延迟和维护要求选工具。
