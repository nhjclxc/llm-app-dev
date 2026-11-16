# **Tokenizer 核心知识点学习路线**

Tokenizer（分词器）是大语言模型（LLM）非常重要的基础组件。

如果你想成为一名大模型应用开发工程师，我建议你认真学习 Tokenizer。因为它直接影响模型的输入、输出、上下文长度、推理成本以及多语言处理能力。

首先要理解一个核心概念：

**大语言模型并不是直接理解文字，而是先通过 Tokenizer 将文本转换成 Token ID，再将 Token ID 转换成 Embedding，输入 Transformer。**

[huggingface 官方 tokenizer 教学](https://huggingface.co/docs/transformers/en/tokenizer_summary)


## **Tokenizer 在 LLM 中的位置**

**原始文本（Text）**

Hello, how are you?

**Tokenizer**

文本切分与编码

**Token IDs**

[15496, 11, 703, 389, 345, 30]

**Embedding Layer**

将离散 Token ID 转换为向量

**Transformer**

Attention、Feed Forward 等

**模型输出 Token IDs**

**Tokenizer Decode**

转换为自然语言文本

我建议将 Tokenizer 分为以下 8 个模块学习。

## **一、Tokenizer 基础概念**

必须掌握

首先需要理解以下概念：

| 知识点             | 含义                   |
| --------------- | -------------------- |
| Token           | 模型处理的基本文本单元          |
| Tokenizer       | 将文本转换为 Token 的工具     |
| Vocabulary      | 词表，保存 Token 与 ID 的映射 |
| Token ID        | 每个 Token 对应的整数编号     |
| Encoding        | 文本编码成 Token ID       |
| Decoding        | Token ID 转换回文本       |
| Special Tokens  | 特殊 Token             |
| Vocabulary Size | 词表大小                 |

例如：

```
text = "I love machine learning"tokens = ["I", " love", " machine", " learning"]token_ids = [40, 3021, 3821, 4673]
```

注意：以上 Token 和 ID 仅用于演示，不是真实模型的词表映射。

你需要理解：

1. 为什么模型不能直接处理字符串？
2. Token 和 Word 有什么区别？
3. 一个英文单词是否一定对应一个 Token？
4. 一个中文汉字是否一定对应一个 Token？
5. Token ID 和 Embedding 有什么关系？
6. Vocabulary Size 为什么会影响 Embedding Layer 的参数量？

例如，假设词表大小为 50,000，Embedding 维度为 768，那么 Embedding 矩阵的参数量就是：

\\[ 50000 \times 768 = 38,400,000 \\]

这意味着词表大小会直接影响模型的参数规模。

## **二、Tokenization 分词算法**

最重要的模块

这是 Tokenizer 最核心的知识点。

你需要重点了解以下几种算法。

### **1. Character-level Tokenization（字符级分词）**

将文本按照字符拆分。

例如：

```
Hello
```

分词结果：

```
["H", "e", "l", "l", "o"]
```

优点：

- 实现简单。
- 不容易遇到未登录词（OOV）。

缺点：

- Token 数量通常较多。
- 序列长度容易增加。
- 难以直接利用词级语义结构。

### **2. Word-level Tokenization（词级分词）**

按照完整单词进行切分。

例如：

```
I love machine learning
```

分词结果：

```
["I", "love", "machine", "learning"]
```

优点：

- Token 数量相对较少。
- 单词语义比较完整。

缺点：

- 词表可能非常庞大。
- 容易遇到 OOV（Out Of Vocabulary）。
- 不方便处理拼写变化和生僻词。

### **3. Subword Tokenization（子词级分词）**

重点掌握

这是现代大语言模型最常见的分词思路。

它介于字符级和词级之间。

例如：

```
unbelievable
```

可能被拆分为：

```
["un", "believ", "able"]
```

它解决了两个问题：

- 不需要为每一个完整单词都建立词表。
- 可以通过组合子词处理大量未见过的单词。

你需要重点学习三种子词算法：

[How LLMs See the World - ByteByteGo Newsletter](https://images.openai.com/static-rsc-4/miWjm5hfxeituPD5_zm0aIxruNlc1DZmo7ph6bQSwm0wKaw2S7S64zSSSqu9pu8eJYkDSc-4c-xEeAIEKOzGWGsVhuulp8K4Z2BMyL4yXthPL5uIfwdRpwAU0jQfsVFuWFhREdqg1M3aQBlEte51emEWVBzvSMItQ5t1jqUWqCM?purpose=inline)

**① BPE（Byte Pair Encoding）**

字节对编码，通过反复合并语料中高频出现的相邻单元，逐步构建词表。

代表：GPT 系列中使用过的 BPE 变体

[Tokenization tradeoff (BPE VS WORDPIECE) | by Databyte | Mar, 2026 | Medium](https://images.openai.com/static-rsc-4/RtH2UQA9lswFy3HCyyQoaloXYt2J97HlHCAGEffQflAYFIP4JgOGHTdrVlVkaxp6afhboGAjPPCWC4tMzvZSl0PMOf5eLCu1iXRJGeVK9pwRvyEvDwiBemFbPPCwoattFciyMWJIUFJ1ISahFbZ9PRJgtB3UM0rXY5f31kKjEGw?purpose=inline)

**② WordPiece**

子词分词算法，根据语言模型目标选择合适的子词组合。

代表：BERT

[Unigram Language Model Tokenization: Probabilistic Subword Segmentation - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](https://images.openai.com/static-rsc-4/NZYVv60ue_0HV2ePyritNMSpyesJCfLK-ZHp-cNJklU3NNPqBKLDf1xKBzAFw-BIcCgU3MhGzt0eT_c82-HKhriD8qYhJ_ozakUiSgZXGSdgTL4jJfkpqn6OrX-R7-3aLEYmA0KXEIY-uaR2KDVYgBFdr52N0h5WRKJOg0Rpdk0?purpose=inline)

**③ Unigram**

基于概率模型，从候选词表中选择合适的子词切分方式。

代表：SentencePiece Unigram

建议掌握程度：

| 算法        | 学习要求             |
| --------- | ---------------- |
| BPE       | 深入理解原理，最好手写实现    |
| WordPiece | 理解原理和 BERT 的使用方式 |
| Unigram   | 理解概率思想和基本流程      |

另外还需要知道 Byte-level BPE 和 SentencePiece。

特别注意：SentencePiece 是一个 Tokenizer 工具及框架，支持 BPE 和 Unigram 等算法，并不等于 Unigram 算法本身。

## **三、Tokenizer 的预处理过程**

必须掌握

Tokenizer 并不一定直接对原始字符串进行分词。

通常还会涉及文本预处理。

主要知识点：

| 知识点                      | 说明           |
| ------------------------ | ------------ |
| Normalization            | 文本规范化        |
| Lowercasing              | 英文字母大小写处理    |
| Unicode                  | Unicode 字符编码 |
| Whitespace               | 空格处理         |
| Pre-tokenization         | 初步切分         |
| Byte Encoding            | 字节编码         |
| Special Token Processing | 特殊 Token 处理  |

例如：

```
Hello, WORLD!
hello, world!
```

某些 Tokenizer 会进行小写规范化，使其更容易共享词表单元。

但是不同模型的处理规则不同，不能认为所有 Tokenizer 都会转小写。

**尤其需要学习 Unicode 和 UTF-8。**

因为中文、英文、emoji 等字符在底层字节表示上存在差异，这会影响 Tokenization 的结果。

## **四、Special Tokens（特殊 Token）**

必须掌握

Special Tokens 是大模型中非常重要的概念。

它们通常承担控制模型输入输出结构的作用。

以常见模型中的特殊 Token 为例：

| Token      | 作用       |
| ---------- | -------- |
| `[PAD]`    | 填充 Token |
| `[UNK]`    | 未知 Token |
| `[CLS]`    | 分类任务起始标记 |
| `[SEP]`    | 分隔标记     |
| `[MASK]`   | 掩码标记     |
| `BOS`      | 序列开始     |
| `EOS`      | 序列结束     |
| `EOT`      | 一轮对话结束   |
| `IM_START` | 消息开始     |
| `IM_END`   | 消息结束     |

注意：这些 Token 不一定同时存在于同一个模型中，而且不同模型的具体命名和语义可能不同。

例如，GPT 风格的 Chat Template 可能长这样：

```
<|im_start|>system
你是一个助手。
<|im_end|>

<|im_start|>user
什么是机器学习？
<|im_end|>

<|im_start|>assistant
机器学习是人工智能的一个分支……
<|im_end|>
```

你需要理解：

- 为什么需要特殊 Token？
- EOS 和 EOT 有什么区别？
- PAD 为什么存在？
- BOS、EOS 在训练中有什么作用？
- 为什么不能随意把普通字符串当成特殊 Token？
- Special Token 的 ID 如何分配？

特别推荐你深入理解 Chat Template，因为它是 Tokenizer 与大模型对话格式之间的重要桥梁。

## **五、Tokenizer 的编码与解码**

必须掌握

这是实际开发中最常用的功能。

使用 Hugging Face Transformers：

```
from transformers import AutoTokenizertokenizer = AutoTokenizer.from_pretrained(    "bert-base-chinese")text = "我正在学习大模型开发"# 编码encoded = tokenizer(    text,    return_tensors="pt")print(encoded)# 查看 Tokentokens = tokenizer.tokenize(text)print(tokens)# 查看 Token IDtoken_ids = tokenizer.encode(text)print(token_ids)# 解码decoded = tokenizer.decode(token_ids)print(decoded)
```

你需要掌握以下 API：

| API                       | 作用             |
| ------------------------- | -------------- |
| `tokenize()`              | 文本切分成 Token    |
| `encode()`                | 文本转换成 Token ID |
| `decode()`                | Token ID 转换成文本 |
| `convert_tokens_to_ids()` | Token 转 ID     |
| `convert_ids_to_tokens()` | ID 转 Token     |
| `get_vocab()`             | 获取词表           |
| `add_tokens()`            | 添加普通 Token     |
| `add_special_tokens()`    | 添加特殊 Token     |
| `save_pretrained()`       | 保存 Tokenizer   |
| `from_pretrained()`       | 加载 Tokenizer   |

还要重点掌握：

```
encoded = tokenizer(    text,    padding=True,    truncation=True,    max_length=128,    return_tensors="pt")
```

这几个参数在实际大模型开发中非常重要。

- `padding`：将不同长度的文本填充到统一长度。
- `truncation`：超过最大长度时截断。
- `max_length`：限制 Token 序列最大长度。
- `return_tensors`：返回 PyTorch 或 TensorFlow 张量。

## **六、Padding、Truncation 和 Attention Mask**

必须掌握

这部分和你之前学习的 PyTorch DataLoader、Batch 训练密切相关。

假设有三条文本：

```
文本 A: [101, 200, 300]
文本 B: [101, 200, 300, 400, 500]
文本 C: [101, 200]
```

它们长度不同，不能直接组成规则的二维 Tensor。

因此需要 Padding：

```
文本 A: [101, 200, 300,   0,   0]文本 B: [101, 200, 300, 400, 500]文本 C: [101, 200,   0,   0,   0]
```

同时生成 Attention Mask：

```
文本 A: [1, 1, 1, 0, 0]文本 B: [1, 1, 1, 1, 1]文本 C: [1, 1, 0, 0, 0]
```

其中：

- `1`：有效 Token。
- `0`：Padding Token。

模型可以利用 Attention Mask 避免关注填充位置。

需要注意：Attention Mask 的具体约定取决于模型和 API；上面是 Hugging Face 常见的二维 padding mask 表示。

重点知识：

1. Padding 的作用。
2. Truncation 的作用。
3. Dynamic Padding。
4. Attention Mask。
5. Left Padding 和 Right Padding。
6. Padding Token 与 EOS Token 的关系。
7. Padding 对训练和推理的影响。

## **七、Tokenizer 与上下文窗口、推理成本的关系**

应用开发必须掌握

这是工程开发中非常重要的知识点。

假设某个模型的上下文窗口为 32K Tokens。

那么：

```
System Prompt
+
User Prompt
+
历史对话
+
RAG 检索内容
+
模型生成内容
```

所有这些内容都会消耗上下文长度。

例如：

| 内容            | Token 数量 |
| ------------- | -------- |
| System Prompt | 1,000    |
| User Prompt   | 2,000    |
| 历史对话          | 5,000    |
| RAG 检索内容      | 10,000   |
| 预留输出空间        | 4,000    |
| 总计            | 22,000   |

因此你需要理解：

- Token 数量与字符数量有什么区别？
- 为什么中文和英文的 Token 消耗可能不同？
- 为什么不同模型处理相同文本的 Token 数量不同？
- Context Window 如何影响 RAG？
- Token 数量如何影响 API 调用成本？
- 如何设计 Token Budget？
- 如何避免上下文超长？

实际项目中，建议在 RAG 检索阶段就计算 Token 数量，而不是等到请求模型时才发现超出上下文窗口。

## **八、Tokenizer 实战项目**

建议你至少完成下面三个项目。

项目 1 · 必做

**Tokenizer 编码与解码实验**

使用 Hugging Face Transformers 加载 BERT、GPT 类模型的 Tokenizer。

实现：

- 中文、英文、数字、特殊符号分词。
- 查看 Token ID。
- 编码和解码。
- 比较不同模型的 Token 数量。

项目 2 · 必做

**手动实现简易 BPE**

使用 Python 和 Counter 实现一个简易 BPE。

实现：

- 统计相邻 Token 频率。
- 合并最高频的相邻 Token。
- 构建词表。
- 编码和解码。

这个项目能帮助你真正理解 BPE，而不只是会调用 API。

项目 3 · 推荐

**RAG Token Budget 管理器**

实现一个简单的文本 Token 预算管理工具。

实现：

- 统计 Prompt Token 数量。
- 统计历史对话 Token 数量。
- 统计 RAG 文档 Token 数量。
- 根据上下文窗口限制自动截断。
- 为模型生成预留 Token 空间。

## **九、最终学习清单：Tokenizer 必学知识点**

我帮你整理成一份可以逐项完成的学习清单。

**Tokenizer 学习进度**

# **0/31**

**1. 基础概念**

Token、Tokenizer、Vocabulary、Token ID

Encoding 和 Decoding

Token 与 Word 的区别

Vocabulary Size 与 Embedding 参数量

**2. 分词算法**

Character-level Tokenization

Word-level Tokenization

Subword Tokenization

BPE 算法及手动实现

WordPiece 算法

Unigram 算法

Byte-level BPE 和 SentencePiece

**3. 文本预处理**

Unicode 与 UTF-8

Normalization

Pre-tokenization

Whitespace 处理

Byte Encoding

**4. 特殊 Token**

PAD、UNK、BOS、EOS

CLS、SEP、MASK

EOT、IM_START、IM_END

Chat Template

**5. 实际 API**

tokenize、encode、decode

convert_tokens_to_ids

get_vocab、add_tokens

add_special_tokens

save_pretrained、from_pretrained

**6. Batch 与上下文**

Padding 和 Truncation

Dynamic Padding

Attention Mask

Left Padding 与 Right Padding

Context Window 与 Token Budget

Token 数量与 API 成本

重置进度

## **十、推荐学习资料**

1. Hugging Face Tokenizers：[官方文档](https://huggingface.co/docs/tokenizers/?utm_source=chatgpt.com)，学习 Tokenizer 原理与实现。
2. Hugging Face Transformers：[Tokenizer API 文档](https://huggingface.co/docs/transformers/main_classes/tokenizer?utm_source=chatgpt.com)，学习实际使用。
3. SentencePiece：[GitHub 官方项目](https://github.com/google/sentencepiece?utm_source=chatgpt.com)，学习 BPE 和 Unigram。
4. [OpenAI Tokenizer 工具](https://platform.openai.com/tokenizer?utm_source=chatgpt.com)：直观观察文本如何被切分成 Token。

**我的学习建议：**

对于你目前的阶段，Tokenizer 不需要花一个月学习。建议用 3～5 天：

- 第 1 天：基础概念、字符级、词级、子词级分词。
- 第 2 天：BPE、WordPiece、Unigram。
- 第 3 天：Special Tokens、Padding、Attention Mask。
- 第 4 天：Hugging Face Tokenizer API 实战。
- 第 5 天：手写简易 BPE，完成 Token Budget 小项目。

掌握这些之后，就可以继续学习 Embedding 了。