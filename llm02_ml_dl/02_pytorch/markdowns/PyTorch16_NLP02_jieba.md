# jieba 中文分词：从基础切词到文本分析

jieba 是常用的 Python 中文分词工具。中文书写通常没有空格标出词边界，例如“我来到北京清华大学”，需要算法判断连续汉字如何组成词。jieba 结合词典、词频和 HMM（隐马尔可夫模型）新词发现等机制，给出一条合理切分路径。

本篇从安装和基础分词开始，讲解精确、全、搜索模式，HMM、用户词典、词性标注、关键词抽取、停用词、词频、传统文本分类和工程实践。每个知识点均配示例代码。分词受词典、版本、语境影响，不存在适用于所有任务的唯一切分。

> jieba 输出的是中文词级片段，不是 BERT/GPT 的子词 token ID。预训练模型应加载配套 tokenizer，不要把 jieba 切词结果直接当作模型 token 序列。

## 1. jieba 解决什么问题

分词将字符串变成 token 列表，便于词频统计、搜索索引、关键词提取和传统分类。分词错误会传递到下游步骤，因此需要检查实际输出。

```text
自然语言处理需要结合上下文
        -> jieba
自然语言 / 处理 / 需要 / 结合 / 上下文
        -> 统计、检索或分类
```

```python
import jieba

text = "自然语言处理需要结合上下文理解词语。"
tokens = list(jieba.cut(text))
print(tokens)
print("/".join(tokens))
```

jieba.cut 返回可迭代对象，jieba.lcut 直接给出列表。长文本若要节省内存，可逐个消费迭代器；需要重复使用结果时可保存列表。

```python
text = "我喜欢自然语言处理，也喜欢机器学习。"
print(list(jieba.cut(text)))
print(jieba.lcut(text))
```

## 2. 安装与版本检查

当前项目依赖已声明 jieba，可同步项目环境；独立环境可用 pip。记录版本便于解释词典或算法升级后的结果变化。

```bash
uv sync
```

```bash
python -m pip install jieba
```

```python
import jieba
print(jieba.__version__)
```

## 3. 中文分词为什么有歧义

词边界没有直接写在中文句子里。“南京市长江大桥”或“研究生命起源”可能存在不同切法，具体边界会改变含义。jieba 使用词典、词频和模型推断较合理的路径，但不具备理解说话者真实意图的能力。

```python
for text in ["南京市长江大桥", "研究生命起源", "乒乓球拍卖完了"]:
    print(text, "->", "/".join(jieba.cut(text)))
```

通用词典可能不含医疗、法律、产品名、人名等领域词。面对专有词，通常需补充用户词典，并用代表性上下文验证。

```python
text = "智算平台正在部署多模态检索系统。"
print("默认:", "/".join(jieba.cut(text)))
```

## 4. 精确模式

精确模式选择一条较完整、较少重叠的切分路径，适合一般文本分析和分类。默认模式通常就是精确模式。

```python
text = "我来到北京清华大学"
print("/".join(jieba.cut(text, cut_all=False)))
```

精确不代表每个词边界都适合业务语料。分词结果应结合任务检查，且训练与推理应固定一致的配置。

```python
sentence = "长春市长春节讲话"
print(jieba.lcut(sentence, cut_all=False))
```

## 5. 全模式

全模式尽量找出句子中所有可能成词的片段，适合探索候选词或某些召回型搜索。它会产生重叠 token，不保证 token 能拼回原文，也不适合作为唯一的序列标注路径。

```python
text = "我来到北京清华大学"
print("/".join(jieba.cut(text, cut_all=True)))
```

全模式 token 更多不等于更准确。用同句对照不同模式：

```python
text = "南京市长江大桥欢迎您"
print("精确:", "/".join(jieba.cut(text, cut_all=False)))
print("全模式:", "/".join(jieba.cut(text, cut_all=True)))
```

## 6. 搜索引擎模式

搜索模式在较完整切分基础上，对长词进一步拆出可能有用的短词，以提升搜索召回。它适合搜索索引，但会引入冗余片段，不宜直接用于要求 token 一一对应的任务。

```python
text = "小明硕士毕业于中国科学院计算所，后来在日本京都大学深造"
print("精确:", "/".join(jieba.cut(text)))
print("搜索:", "/".join(jieba.cut_for_search(text)))
```

| 模式 | 行为 | 用途 | 注意 |
| --- | --- | --- | --- |
| 精确 | 选择较完整的一条路径 | 分类、一般分析 | 领域边界可能不合适 |
| 全模式 | 返回更多可能词片段 | 候选召回、规则探索 | 有重叠 |
| 搜索 | 对长词补充细粒度词 | 搜索索引 | 召回导向、可能冗余 |

```python
text = "中华人民共和国中央人民政府"
for name, tokens in [
    ("精确", jieba.cut(text, cut_all=False)),
    ("全模式", jieba.cut(text, cut_all=True)),
    ("搜索", jieba.cut_for_search(text)),
]:
    print(name, list(tokens))
```

## 7. HMM 与新词发现

精确模式默认可使用 HMM（Hidden Markov Model，隐马尔可夫模型）识别词典外的人名或新词。它根据字符序列中 B/M/E/S 等状态概率推断边界，能补充词典，但不是可靠的命名实体识别器。

```python
text = "他来到了网易杭研大厦"
print("HMM 开启:", "/".join(jieba.cut(text, HMM=True)))
print("HMM 关闭:", "/".join(jieba.cut(text, HMM=False)))
```

关闭 HMM 可观察词典路径，也可能让未登录词切得更差。是否启用应由验证数据决定，而不是只针对一条例句调参。

```python
sentence = "小明硕士毕业后研究人工智能"
for use_hmm in (True, False):
    print(use_hmm, jieba.lcut(sentence, HMM=use_hmm))
```

新词发现只决定可能的边界，不会自动判断片段是人名、组织还是地点。

```python
text = "李雷加入星河智算公司"
print(jieba.lcut(text, HMM=True))
```

## 8. tokenize 与字符偏移

jieba.tokenize 可同时返回 token 起始和结束位置，适合高亮搜索结果或映射回原文。偏移通常按 Python 字符串索引，右边界为开区间。

```python
text = "我喜欢自然语言处理"
for token, start, end in jieba.tokenize(text):
    print(token, start, end, repr(text[start:end]))
```

搜索模式可能产生重叠 token，因此对应的原文区间也可能重叠。

```python
text = "小明硕士毕业于中国科学院计算所"
for item in jieba.tokenize(text, mode="search"):
    print(item)
```

## 9. 独立 Tokenizer 对象

模块级 jieba.cut 使用默认的全局 tokenizer。可创建 jieba.Tokenizer 实例隔离不同领域配置，避免某个任务加词影响另一个任务。

```python
general = jieba.Tokenizer()
domain = jieba.Tokenizer()
domain.add_word("星河智算平台", freq=10000)
text = "星河智算平台发布了新产品"
print(general.lcut(text))
print(domain.lcut(text))
```

Tokenizer 词典初始化有成本，应复用对象，而不是每处理一条文本就重新创建。

```python
tokenizer = jieba.Tokenizer()

def cut_many(texts):
    return [tokenizer.lcut(text) for text in texts]

print(cut_many(["我喜欢自然语言处理", "机器学习很有趣"]))
```

## 10. 动态加词与删词

add_word 可把领域词加入当前词典。词频会影响路径选择；它不是置信度百分比，数值需要结合上下文验证。del_word 可删除动态添加的词。

```python
text = "星河智算平台正在开发云端推理引擎"
print("原始:", jieba.lcut(text))
jieba.add_word("星河智算平台", freq=10000)
jieba.add_word("云端推理引擎", freq=10000)
print("加词:", jieba.lcut(text))
```

若词性标注也会使用此词，可设置 tag。加词只是更新词典，不会训练新的深度模型。

```python
jieba.add_word("星河智算公司", freq=10000, tag="nt")
print(jieba.lcut("星河智算公司发布模型"))
```

## 11. 用户词典

领域词较多时应集中放在可版本管理的用户词典中，而非散落在代码里。常见文本格式是一行一个词，可带词频和词性：

```text
星河智算平台 10000 nz
云端推理引擎 8000 nz
大语言模型 12000 n
```

加载词典并检查输出：

```python
import jieba

jieba.load_userdict("userdict.txt")
print(jieba.lcut("星河智算平台使用云端推理引擎"))
```

相对路径依赖当前工作目录。Notebook、脚本和服务进程的工作目录可能不同，应构造明确路径并处理文件不存在情况。

```python
from pathlib import Path

path = Path("data/userdict.txt")
if path.exists():
    jieba.load_userdict(str(path))
else:
    print("未找到用户词典:", path.resolve())
```

也可以从文件类对象加载词典，适合小型动态配置；大型词典推荐独立文件。

```python
from io import StringIO

dictionary = StringIO("星河智算平台 10000 nz\n云端推理引擎 8000 nz\n")
jieba.load_userdict(dictionary)
print(jieba.lcut("星河智算平台和云端推理引擎"))
```

## 12. 词频和 suggest_freq

jieba 根据词典词频和候选路径选择整体切分。提高某词频率会增加它作为整体出现的倾向，但上下文中的其他候选也会影响结果。

```python
text = "创新药研发中心"
print("修改前:", jieba.lcut(text))
jieba.add_word("创新药", freq=20000)
jieba.add_word("研发中心", freq=20000)
print("修改后:", jieba.lcut(text))
```

suggest_freq 可微调某个词或切分组合的频率倾向，适合少量规则修正；大量领域规则应维护用户词典。全局频率修改会影响后续调用。

```python
sentence = "如果放到post中将出错。"
print("调整前:", jieba.lcut(sentence, HMM=False))
jieba.suggest_freq(("中", "将"), True)
print("调整后:", jieba.lcut(sentence, HMM=False))
```

```python
word = "台中"
print("原切分:", jieba.lcut(word, HMM=False))
jieba.suggest_freq(word, True)
print("调频后:", jieba.lcut(word, HMM=False))
```

## 13. 词性标注

分词只给边界，词性标注再给 token 赋予名词、动词等标签。jieba.posseg 提供轻量词性标注，不等价于完整句法分析。

```python
import jieba.posseg as pseg

text = "小明在北京学习自然语言处理"
for word, flag in pseg.cut(text):
    print(word, flag)
```

lcut 可直接返回列表，标签字段通常为 item.word 和 item.flag。标签体系是工具约定，项目应按自身 schema 检查和映射。

```python
words = pseg.lcut("苹果公司发布新手机")
print([(item.word, item.flag) for item in words])
```

用户词典中的词性字段可影响领域词标注结果：

```python
jieba.add_word("星河智算公司", freq=10000, tag="nt")
print([(item.word, item.flag) for item in pseg.cut("星河智算公司发布模型")])
```

## 14. TF-IDF 关键词提取

jieba.analyse.extract_tags 使用 TF-IDF 思想为文档提取关键词。topK 控制返回数量，withWeight=True 同时返回分数。分数是算法下的相对权重，不是事实置信度。

```python
import jieba.analyse

document = """
人工智能正在改变软件开发。自然语言处理和机器学习可以分析文本，
大语言模型也推动了智能问答和文本生成的发展。
"""
keywords = jieba.analyse.extract_tags(document, topK=5, withWeight=True)
print(keywords)
```

可用词性过滤候选词。短文本可能缺少足够上下文，结果应结合任务检查。

```python
keywords = jieba.analyse.extract_tags(
    document, topK=10, withWeight=True, allowPOS=("n", "vn", "v")
)
print(keywords)
```

## 15. TextRank 关键词提取

TextRank 将词作为图节点，把窗口内共现词连接，再通过类似 PageRank 的思想排序。它利用局部共现关系，与 TF-IDF 的词频/文档频率方法不同。短文档结果可能不稳定。

```python
keywords = jieba.analyse.textrank(document, topK=8, withWeight=True)
print(keywords)
```

也可限制词性；比较两种方法时，应检查候选词和领域停用词配置。

```python
keywords = jieba.analyse.textrank(
    document, topK=8, withWeight=True, allowPOS=("ns", "n", "vn", "v")
)
print(keywords)
```

## 16. 停用词与清洗

停用词是任务中认为区分度较低的高频词。删除“的”等词有时可缩小特征空间，但否定词“不”“没”对情感判断很重要。停用词表不是普遍正确的标准，需按任务维护。

仓库中有中文停用词表示例：data/dl/tokenizer/stopwords.zh.all.txt。

```python
from pathlib import Path

stop_path = Path("data/dl/tokenizer/stopwords.zh.all.txt")
if stop_path.exists():
    stop_words = {
        line.strip() for line in stop_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
else:
    stop_words = {"的", "了", "和", "是", "在"}

text = "我不喜欢这个产品，它的质量很好。"
tokens = jieba.lcut(text)
filtered = [word for word in tokens if word not in stop_words]
print(tokens)
print(filtered)
```

标点、数字、URL、emoji 和英文缩写是否删除，要看任务需求。下面保留包含汉字、字母或数字的 token：

```python
import re

text = "版本 V2.1 很好用！准确率达到 98%。"
tokens = jieba.lcut(text)
useful = [t for t in tokens if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", t)]
print(tokens)
print(useful)
```

## 17. 词频统计

分词并清洗后可用 Counter 统计词频。高频不等同于重要，建议同时比较不同类别或文档。

```python
from collections import Counter

texts = [
    "自然语言处理需要数据和模型",
    "机器学习模型可以处理文本数据",
    "中文分词帮助处理自然语言",
]
tokens = [word for text in texts for word in jieba.lcut(text)]
print(Counter(tokens).most_common(10))
```

组合停用词过滤和简单内容过滤：

```python
import re
from collections import Counter

all_tokens = [word for text in texts for word in jieba.lcut(text)]
content_words = [
    word for word in all_tokens
    if word not in stop_words and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", word)
]
print(Counter(content_words).most_common())
```

## 18. N-gram 与词袋

N-gram 是连续 n 个 token 的组合。Unigram 统计词，bigram 统计相邻词对。它保留有限局部顺序，但组合会稀疏增长。

```python
tokens = jieba.lcut("自然语言处理需要上下文")
bigrams = list(zip(tokens, tokens[1:]))
print(tokens)
print(bigrams)
```

词袋把文档变成词项计数向量，不保留顺序。jieba 负责切词，scikit-learn 可负责词袋/TF-IDF 和分类器。

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["这部电影非常精彩", "剧情紧凑演员出色"]
vectorizer = CountVectorizer(tokenizer=jieba.lcut, token_pattern=None)
X = vectorizer.fit_transform(texts)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

## 19. TF-IDF 中文文本分类基线

传统分类流程是 jieba 切词、TF-IDF 向量化、线性分类器学习标签。传入自定义 tokenizer 时关闭默认 token_pattern，避免 scikit-learn 再用默认规则切分。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def chinese_tokenizer(text):
    return [word for word in jieba.lcut(text) if word.strip()]

texts = ["这部电影非常精彩", "这部电影一点也不好看", "剧情紧凑演员出色", "剧情混乱表演糟糕"]
vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None)
X = vectorizer.fit_transform(texts)
print(vectorizer.get_feature_names_out())
print(X.shape)
```

训练/测试切分后，只在训练集 fit 词表和 IDF，测试集只 transform。下面演示完整小型分类基线：

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "这部电影非常精彩", "这部电影一点也不好看",
    "剧情紧凑演员出色", "剧情混乱表演糟糕",
    "故事温暖值得推荐", "故事拖沓不值得观看",
]
labels = [1, 0, 1, 0, 1, 0]
train_x, test_x, train_y, test_y = train_test_split(
    texts, labels, test_size=0.33, random_state=42, stratify=labels
)
model = make_pipeline(
    TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None),
    LogisticRegression(max_iter=1000),
)
model.fit(train_x, train_y)
print("test accuracy:", model.score(test_x, test_y))
```

## 20. 文件与批量语料

读取文件时显式指定编码。每篇文档单独分词并保留边界，避免 n-gram 意外跨文档。

```python
from pathlib import Path

path = Path("data/dl/tokenizer/stopwords.zh.all.txt")
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines()[:5]:
        print(line, jieba.lcut(line))
```

```python
documents = ["我喜欢自然语言处理。", "机器学习也很有趣。"]
tokenized_documents = [jieba.lcut(doc) for doc in documents]
for doc_id, tokens in enumerate(tokenized_documents):
    print(doc_id, tokens)
```

长文件可以逐行读取，减少一次载入全部文本的内存使用：

```python
from pathlib import Path

path = Path("large_corpus.txt")
if path.exists():
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            tokens = jieba.lcut(line)
            if line_number <= 3:
                print(line_number, tokens)
```

## 21. 并行分词

jieba 提供并行分词接口，适合支持的系统环境中处理大型文件。它基于多进程，短文本和在线请求不适合为每次调用启动进程；部署前按平台测试，并避免并行时动态修改共享词典。

```python
# 只在已准备好大型文本文件且当前平台支持时运行
jieba.enable_parallel(4)
try:
    with open("large_corpus.txt", encoding="utf-8") as source:
        for tokens in jieba.cut(source.read()):
            pass
finally:
    jieba.disable_parallel()
```

多进程启动有开销，实际是否加速应基准测试。逐行读取通常更容易控制内存：

```python
from pathlib import Path

path = Path("large_corpus.txt")
if path.exists():
    with path.open(encoding="utf-8") as source:
        for line in source:
            tokens = jieba.lcut(line)
```

## 22. 可复现与工程实践

模型训练与线上推理必须使用相同的 jieba 版本、词典、HMM 设置、用户词典、规范化和停用词策略。只保存分类模型而不保存文本处理配置，会造成训练/推理特征不一致。

```python
preprocess_config = {
    "jieba_version": jieba.__version__,
    "mode": "precise",
    "HMM": True,
    "userdict": "data/userdict.txt",
    "stopwords": "data/dl/tokenizer/stopwords.zh.all.txt",
}
print(preprocess_config)
```

对关键领域词建立回归样例。升级 jieba 或词典后比较输出，而不是只看程序是否还能运行。

```python
tokenizer = jieba.Tokenizer()
actual = tokenizer.lcut("自然语言处理")
print(actual)
# 确认目标边界后，可在项目测试中断言：
# assert actual == ["自然语言", "处理"]
```

相对路径应基于明确项目目录构造。服务初始化时加载词典并复用 tokenizer，不要在每个请求里修改全局词典。

```python
from pathlib import Path

project_root = Path.cwd()
userdict_path = project_root / "data" / "userdict.txt"
print(userdict_path.resolve())
```

## 23. jieba 与 Transformer tokenizer 的区别

jieba 主要以词典和中文词边界输出词语，适合传统搜索、词频分析和词袋特征。预训练 Transformer 使用自己的子词/字节 tokenizer，ID 与模型 embedding 行绑定，必须配套使用。

```python
# 传统特征路径
print(jieba.lcut("自然语言处理"))

# Transformer 路径需加载模型自己的 tokenizer（示意）
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("model-name")
# encoded = tokenizer("自然语言处理", return_tensors="pt")
# print(encoded["input_ids"])
```

如果微调中文预训练模型，默认遵循其 tokenizer；除非模型训练约定明确要求，否则不要预先用 jieba 切词。

## 24. 常见问题

| 表现 | 可能原因 | 检查 |
| --- | --- | --- |
| 专有名词被切碎 | 通用词典缺少领域词 | 添加用户词典并测试多个上下文 |
| 词被错误合并 | 词频倾向/领域边界不合适 | 检查 suggest_freq，避免盲目调频 |
| 关键词被常用词占据 | 停用词或语料规模不合适 | 审核停用词，比较 TF-IDF/TextRank |
| 同文输出变化 | 版本、词典、HMM 配置变化 | 固定依赖和配置 |
| 中文模型效果变差 | 把 jieba 当作模型 tokenizer | 加载模型配套 tokenizer |
| 文件乱码 | 编码不一致 | 显式指定 encoding 并检查源文件 |
| 加词似乎无效 | 频率、上下文或 tokenizer 对象不一致 | 核对实例、词典和实际输出 |

用同一文本切换 HMM 和模式有助于缩小问题范围：

```python
text = "星河智算平台发布了新模型"
print(jieba.__version__)
print(jieba.lcut(text, cut_all=False, HMM=True))
print(jieba.lcut(text, cut_all=False, HMM=False))
```

## 25. 推荐练习

按“观察 -> 切词 -> 词典 -> 特征 -> 评估”的顺序练习，每一步都记录配置与中间结果。

```python
exercises = [
    "比较 cut、lcut 和 cut_for_search",
    "对照精确、全模式、搜索模式和 HMM 开关",
    "给组织名和产品名建用户词典并加回归测试",
    "检查停用词表中的否定词是否应保留",
    "比较 TF-IDF 与 TextRank 关键词",
    "搭建 jieba + TF-IDF + Logistic Regression 分类基线",
]
for number, exercise in enumerate(exercises, 1):
    print(number, exercise)
```

## 26. 总结

jieba 的基本路径是：选择合适模式、检查输出、按领域维护词典、按任务处理停用词和清洗，再做统计、检索或传统特征建模。HMM、词频调整、偏移、词性、关键词算法和并行属于进一步能力。

```text
原文
  -> jieba 分词（模式、HMM、用户词典）
  -> 任务相关过滤
  -> 词频、关键词、搜索索引或 TF-IDF 特征
  -> 模型训练与独立评估
```

分词是算法对边界的判断，不是无误的语言真值；不同模式服务不同任务；词级 jieba 不能取代 Transformer 配套 tokenizer。
