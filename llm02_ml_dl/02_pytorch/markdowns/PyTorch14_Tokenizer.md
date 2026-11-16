# Tokenizer 分词：从文本到模型输入

Tokenizer（分词器）把文本转换为模型可处理的 token ID 序列，也可将 ID 解码回文本。它不只是按空格切词：规范化、预切分、子词算法、特殊 token、截断、填充和 mask 都会影响模型实际接收到的输入。

本文由浅入深介绍字符级、词级、子词级、BPE、WordPiece、Unigram、预处理 pipeline、特殊 token、编码解码、批处理、Chat Template、上下文窗口与推理成本。示例以 Python、PyTorch 2.x 和 Hugging Face 常见 API 为例；具体行为应以目标模型的 tokenizer 配置为准。

> 重要原则：模型权重必须与训练时使用的 tokenizer、词表、特殊 token 定义和 ID 映射匹配。词表大小相同不代表 ID 含义相同。

## 1. 从文本到模型输入

    原始文本 -> 规范化 -> 预切分 -> 子词切分/查词表 -> token ID -> 添加结构 token -> 截断/填充/mask -> embedding -> 模型

神经网络通常不接收字符串，而接收整数序列 [t1, t2, ..., tL]。Embedding 按 ID 查出向量，Transformer 对序列计算上下文表示。Tokenizer 决定模型看到的基本符号，也决定序列长度。

    text = "I like apples."
    tokens = ["I", "like", "apples", "."]
    vocab = {token: i for i, token in enumerate(tokens)} # vocabulary
    input_ids = [vocab[token] for token in tokens]
    print(input_ids)

## 2. Token、词表与 ID

- Token：字符、单词、子词、字节片段或结构标记。
- Vocabulary（词表）：token 到整数 ID 的映射。
- Token ID：embedding 表索引，没有大小或语义关系。
- 序列长度 L：token 个数，不等于字符数或空格分隔的单词数。

    vocab = {"[UNK]": 0, "cat": 1, "##s": 2}
    ids = [vocab["cat"], vocab["##s"]]
    print(ids)  # [1, 2]

不同 tokenizer 的 ID 不能混用。ID 20 不比 ID 10 更“大”，它只是另一个 embedding 行。

## 3. 字符级分词

字符级 tokenizer 把文本拆成字符，也可能按 Unicode code point、字节或 grapheme cluster 切分。它词表小、较容易覆盖新词，但序列长且单个字符语义弱。

    text = "你好，世界!"
    tokens = list(text)
    char_to_id = {ch: i for i, ch in enumerate(sorted(set(tokens)))}
    ids = [char_to_id[ch] for ch in tokens]
    print(tokens, ids)

Python 的 list(text) 是按 Unicode code point 切分，不一定等于用户感知的字符；组合音标和 emoji 可能由多个 code point 组成。字符级适合教学或特定任务，现代语言模型更常使用子词或字节级方法。

## 4. 词级分词

词级方法按完整词或规则切分。英语常利用空格和标点；中文没有天然空格边界，需要语言规则或中文分词器。有限词表会使未登录词变成 UNK。

    import re
    text = "Tokenizer is useful, isn't it?"
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    print(tokens)

词级 token 直观、序列通常短，但词表易膨胀，词形变化、拼写变化和新词会造成 OOV。lower() 会损失大小写，不能对所有任务无条件使用。

## 5. 子词分词

子词（subword）让常见片段成为一个 token，罕见词拆成多个片段。例如 unhappiness 可能拆为 un、happi、ness，实际结果取决于词表。它折中词表规模、序列长度与 OOV 覆盖。

    词级：   ["unhappiness"] 或 [UNK]
    字符级： ["u", "n", "h", ...]
    子词级： ["un", "happi", "ness"]

子词 tokenizer 通常需要能覆盖任意输入的基础单位，如字符或 byte-level 字母表。具体覆盖范围由算法和配置决定。

## 6. BPE

Byte Pair Encoding（BPE）应用于 NLP 时通常从字符/字节单位开始，反复合并语料中最常见的相邻 token 对，直至达到词表大小或合并次数。训练得到的 merge 规则用于编码新文本。

    from collections import Counter
    words = ["l o w", "l o w e r", "n e w e s t"]
    counts = Counter()
    for word in words:
        symbols = word.split()
        counts.update(zip(symbols, symbols[1:]))
    print(counts.most_common(1))

这是合并思想的教学演示，不是完整 BPE。真实实现还涉及预切分、词频、边界、字节表示、停止条件和高效结构。Byte-level BPE 也是常见变体，不能只凭 BPE 名称推断空格、标点或 Unicode 行为。

### 6.1 从零实现可训练 BPE

BPE 要区分训练与编码两个阶段：

1. 把每个词表示为字符序列，并在末尾追加词尾标记。例如 lower 变为 l、o、w、e、r、词尾标记。
2. 用词频加权统计整个语料中相邻 pair 的频次。
3. 选择频次最高的 pair，合并成新符号，并按先后顺序记录 merge rule。
4. 对合并后的语料重复统计，直至达到 num_merges 或没有可合并的 pair。
5. 编码新词时，先按相同的基础符号拆分，再反复应用训练顺序最靠前的可用 merge。

词尾标记避免不同词之间错误合并，也区分词尾与词内的相同字符序列。pair 频次相同时必须设定稳定的平局规则，否则相同语料可能训练出不同结果。

仓库中的 bpe_from_scratch.py 是只依赖 Python 标准库的完整教学实现。核心 pair 合并逻辑如下：

    def merge_pair(pair, word):
        merged, index = [], 0
        while index < len(word):
            if index + 1 < len(word) and (word[index], word[index + 1]) == pair:
                merged.append(word[index] + word[index + 1])
                index += 2
            else:
                merged.append(word[index])
                index += 1
        return tuple(merged)

训练阶段必须用词频作为权重，否则低频词与高频词对 pair 的影响会相同：

    from collections import Counter

    def pair_frequencies(word_counts):
        frequencies = Counter()
        for word, count in word_counts.items():
            for pair in zip(word, word[1:]):
                frequencies[pair] += count
        return frequencies

使用完整实现：

    from bpe_from_scratch import SimpleBPE

    corpus = ["low lower newest", "low lowest newer", "low lower"]
    bpe = SimpleBPE(num_merges=20).fit(corpus)
    tokens = bpe.encode("lowest newer")
    print(tokens)
    print(bpe.decode(tokens))

本实现使用空白预切分和字符初始词表，只用于理解算法。工业实现还必须处理 Unicode 规范化、byte-level 覆盖、标点及空格保留、special token、词表 ID 序列化、offset mapping 和大语料性能。

## 7. WordPiece

WordPiece 与 BPE 一样构造子词词表，但常见描述是依据语言模型目标或 likelihood 增益选择合并，而不是只选频率最高的相邻对。编码通常使用最长匹配；词中片段可能带 ## 前缀，例如 playing -> play + ##ing。

    词表示意：play, ##ing, ##ed
    playing -> ["play", "##ing"]

双井号（##）是部分实现使用的词内边界标记，不是所有 tokenizer 都有。BERT 常用 WordPiece，但具体大小写、中文和清理规则仍应查看配置。

## 8. Unigram 与 SentencePiece

Unigram 从较大的候选子词集合开始，为 token 估计概率，逐步删除对语料 likelihood 影响较小的候选，直至目标词表大小。编码选择概率较高的切分，也可在训练时采样不同切分做数据增强。

    "lower" 的候选切分可包括 ["lower"]、["low", "er"]、["l","o","w","e","r"]
    Unigram 根据子词和整句概率选择或采样切分。

BPE/WordPiece 通常通过合并或贪心规则确定切分；Unigram 对候选切分建模概率。SentencePiece 是 tokenizer 工具框架，不等于某一种算法，它可训练 Unigram 或 BPE。

| 方法 | 词表构建直觉 | 常见编码 | 注意点 |
| --- | --- | --- | --- |
| BPE | 合并常见相邻单位 | 应用 merge 规则 | 变体和预切分影响结果 |
| WordPiece | 优化语料模型目标 | 常用最长匹配 | ## 是实现约定 |
| Unigram | 概率建模候选集合并剪枝 | 概率最优或采样 | 可能有多种合理切分 |

## 9. 预处理 Pipeline

### 9.1 Normalization（规范化）

规范化可包含 Unicode 规范化、大小写转换、去重音、空白处理和控制字符清理。可能造成信息损失；代码、姓名、日期及大小写敏感任务应谨慎选择。

    import unicodedata
    text = "Cafe\u0301"
    normalized = unicodedata.normalize("NFC", text)
    print(normalized)

NFC 与 NFKC 不等价，NFKC 会更激进地折叠兼容字符。不要无条件 lower() 或 NFKC。

### 9.2 Pre-tokenization（预切分）

预切分在子词算法之前划定候选边界，可能按空格、标点、数字或正则处理，影响可合并片段及空格编码。

    import re
    pieces = re.findall(r"\w+|[^\w\s]", "Hello, world! 42")
    print(pieces)

这是教学正则，不适用于多语言生产。模型作者通常定义了更完整的预切分规则。

### 9.3 Model、Post-processor 与 Decoder

Model 部分使用 BPE、WordPiece、Unigram 或 WordLevel 词表生成 token。Post-processor 可添加 [CLS]、[SEP]、<s>、</s> 等结构 token。Decoder 将 token 还原为文本，负责空格拼接、标点清理和特殊 token 处理。

    BERT 单句示意：[CLS] sentence [SEP]
    句对示意：     [CLS] A [SEP] B [SEP]

这里只是部分 encoder 模型的约定，不能套用于所有模型。经过规范化、截断或跳过特殊 token 后，decode 不一定逐字符可逆。

## 10. 特殊 Token

| 类别 | 例子 | 作用 |
| --- | --- | --- |
| 未知 | [UNK] | 无法覆盖输入时回退 |
| 填充 | [PAD] | 让 batch 序列等长 |
| 序列边界 | [BOS]、[EOS] | 开始/结束，依模型而定 |
| 分类/分隔 | [CLS]、[SEP] | 部分 encoder 模型结构标记 |
| 对话角色 | <|user|>、<|assistant|> | 角色与消息边界 |
| 工具控制 | tool call 标记 | 结构化调用与控制流 |

字符串长得像 special token，不代表 tokenizer 已将它注册为不可拆分的标记。检查实际配置和 ID。

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(tokenizer.pad_token, tokenizer.pad_token_id)
    print(tokenizer.cls_token, tokenizer.cls_token_id)
    print(tokenizer.sep_token, tokenizer.sep_token_id)

新增 token 时必须同步扩展模型 embedding，并训练新行；只更新 tokenizer 可能造成 ID 越界。

    from transformers import AutoTokenizer, AutoModel
    name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|domain|>"]}
    )
    if added:
        model.resize_token_embeddings(len(tokenizer))

新增后成套保存 tokenizer 和模型；新 embedding 通常随机初始化，需要训练来学习含义。

## 11. 编码与解码

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "Tokenizers turn text into IDs."
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    encoded = tokenizer(text, add_special_tokens=True)
    print(tokens)
    print(ids)
    print(encoded["input_ids"])
    print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))

tokenize() 往往只切正文；完整调用 tokenizer(...) 还可添加 special tokens、截断、padding、mask 并返回张量。

    ids = tokenizer("A small example.")["input_ids"]
    print(tokenizer.decode(ids, skip_special_tokens=False))
    print(tokenizer.decode(ids, skip_special_tokens=True))

跳过 special token 适合提取自然语言，但调试结构化生成时不要丢弃控制标记。

## 12. Padding、Truncation 与 Attention Mask

batch Tensor 需要共同长度。Padding 用 PAD 补短序列；truncation 在超过模型上限时裁剪。截断可能删掉任务信息，应选择保留方向、句对策略或分块方法。

常见 tokenizer 的 attention_mask 中 1 表示有效 token、0 表示 PAD。它一般屏蔽 padding，不等同于自回归模型禁止看未来的 causal mask。

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    batch = tokenizer(
        ["short text", "this is a somewhat longer sentence"],
        padding=True,
        truncation=True,
        max_length=12,
        return_tensors="pt",
    )
    print(batch["input_ids"].shape)
    print(batch["attention_mask"])  # 1=有效，0=PAD

固定长度 padding 便于静态 shape 但可能浪费计算；动态 padding 到 batch 最长样本减少浪费。pad_to_multiple_of=8/16 有时有利于硬件优化，仍要遵守最大长度。

    batch = tokenizer(
        ["one", "a longer example"],
        padding="longest",
        truncation=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

Decoder-only 生成时 left padding 可能便于对齐最后有效 token，但必须按模型和 generation 实现确认；encoder 常见 right padding。不同 PyTorch attention API 的布尔 mask 语义可能相反。nn.MultiheadAttention 的 key_padding_mask 中 True 表示忽略该 key。

    valid = batch["attention_mask"].bool()
    key_padding_mask = ~valid

loss 的 ignore_index=pad_id 只忽略目标位置 PAD 的 loss，不会自动屏蔽输入 PAD。

## 13. 训练 BPE 与加载预训练 Tokenizer

以下用 Hugging Face tokenizers 展示训练小型 BPE。真实项目要配置适合任务的 normalizer、pre-tokenizer、post-processor 和 special tokens。

    # pip install tokenizers
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=2000, min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    )
    tok.train_from_iterator(
        ["small corpus for tokenizer", "learn subword units from text"],
        trainer=trainer,
    )
    print(tok.encode("subword tokenizer").tokens)
    tok.save("tokenizer.json")

应用预训练模型时，优先加载同 checkpoint tokenizer。Fast tokenizer 通常支持 offset mapping，可将 token 对齐回原文字跨度。

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    result = tok("Tokenizer example.", return_offsets_mapping=True)
    print(result.tokens(), result["offset_mapping"])

Offset 可用于 NER、抽取式 QA 和高亮；规范化或 byte-level 下需按实现理解。特殊 token 的 offset 常是 (0, 0)。

## 14. 句对与 Token Type IDs

有些 encoder tokenizer 支持句对输入并返回 token_type_ids 区分 A/B；不是每个模型都使用或返回该字段。

    pair = tok(
        "question text", "candidate passage",
        padding=True, truncation="longest_first",
        max_length=128, return_tensors="pt",
    )
    print(pair.keys(), pair["input_ids"].shape)

长句对可选择 longest_first、only_first、only_second 等策略，并确认 tokenizer 支持。

## 15. Chat Template

聊天模型输入通常需模型专属模板，包含角色、消息边界、通道、工具调用和结束标记，不能随意拼角色名与正文。

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain tokenization briefly."},
    ]
    chat_ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    )
    print(chat_ids.shape)

Tokenizer 必须支持该模板。训练、推理模板与停止条件需一致；工具调用和多模态标记可能代表控制流。模板缺失时查看模型卡或 tokenizer.chat_template，不要猜测。

## 16. 上下文窗口与推理成本

上下文窗口以 token 计，不以字符、单词或页面计。预算还要扣除系统提示、历史、工具定义、模板特殊 token，并预留生成长度。

    prompt = "Summarize the conversation."
    ids = tok(prompt, add_special_tokens=True)["input_ids"]
    prompt_tokens = len(ids)
    max_context = 8192
    max_new_tokens = 1000
    print(prompt_tokens, max_context - max_new_tokens)

8192 窗口预留 1000 新 token，prompt 理论上限约 7192，还需容纳模板与框架开销。中文字符不保证一字一个 token，英文单词也可能被拆开，要用目标 tokenizer 实测。

    for text in ["Hello, world!", "你好，世界！", "def f(x): return x + 1"]:
        print(len(tok(text, add_special_tokens=False)["input_ids"]), text)

标准全注意力 Transformer 的 prefill 注意力计算随长度 L 大致 O(L^2)，投影/MLP 部分大致随 L 线性增长。自回归生成每步处理新 token；KV cache 避免重算历史，但其占用近似随层数、batch、上下文长度和 KV head 数线性增长。输出 token 越多，decode 步数越多。

KV cache 粗略内存估算（字节）：

    2 × layers × batch × sequence_length × kv_heads × head_dim × bytes_per_value

系数 2 对应 Key 和 Value。量化、分页 cache、共享前缀和并行实现会影响实际值。裁剪重复上下文、分块、长度分桶和限制 max_new_tokens 可降低成本。

## 17. 子词与词标签对齐

NER 等标签常按原始词标注，一个词可能拆成多个子词。常见策略：只给首子词标签、向全部子词复制标签，或后续子词 loss label 设为 -100。训练与评估规则必须一致。

    text = "Hugging Face builds tools."
    result = tok(text, return_offsets_mapping=True)
    for token, (start, end) in zip(result.tokens(), result["offset_mapping"]):
        print(token, repr(text[start:end]))

Fast tokenizer 的 word_ids() 可将子词对应回原始词；special token 通常没有文本跨度。

## 18. Tokenizer 与模型训练

- 从头训练：先确定 tokenizer，再按 ID 范围创建 embedding 和输出层。
- 微调：沿用 checkpoint 配套 tokenizer。
- 新增 token：扩展 embedding 并训练新增行。
- 部署：成套保存模型、tokenizer、配置和 special token IDs。
- 批处理：核对 PAD、attention mask、label mask 与模型配置。

    from transformers import AutoTokenizer, AutoModel
    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    print(len(tok), model.get_input_embeddings().num_embeddings)
    print(tok.pad_token_id, model.config.pad_token_id)

有些模型会留额外 embedding 行作硬件对齐，数字比较只能用于诊断，不能单独证明 tokenizer 匹配；还要核对 checkpoint 与特殊 ID。

## 19. 教学级词级 Tokenizer

下面演示建词表、编码、未知词回退、解码，不支持复杂 Unicode、子词或精确恢复空格。

    import re
    class TinyWordTokenizer:
        def __init__(self, texts):
            self.token_to_id = {"[PAD]": 0, "[UNK]": 1}
            for text in texts:
                for token in re.findall(r"\w+|[^\w\s]", text.lower()):
                    if token not in self.token_to_id:
                        self.token_to_id[token] = len(self.token_to_id)
            self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        def encode(self, text):
            tokens = re.findall(r"\w+|[^\w\s]", text.lower())
            return [self.token_to_id.get(t, 1) for t in tokens]
        def decode(self, ids):
            return " ".join(self.id_to_token.get(i, "[UNK]") for i in ids)

    toy = TinyWordTokenizer(["I like cats.", "Cats like naps."])
    ids = toy.encode("I like robots.")
    print(ids, toy.decode(ids))

教学实现用空格拼回文本会破坏标点间距。生产 tokenizer 还需稳定序列化、规范化、special token、边界规则与高效批处理。

## 20. DataLoader 动态 Padding

在 collate_fn 中对当前 batch 动态 padding，可避免所有样本都补到全局最大长度。

    from torch.utils.data import DataLoader
    texts = ["short", "a longer sentence for batching"]
    def collate_texts(examples):
        return tok(
            examples, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
    loader = DataLoader(texts, batch_size=2, collate_fn=collate_texts)
    batch = next(iter(loader))
    print(batch["input_ids"].shape, batch["attention_mask"].shape)

大型数据集可预先 tokenize 并缓存以减少重复 CPU 时间，也可训练时处理以省磁盘并支持动态增强；需权衡吞吐、数据变化和可复现性。

## 21. 常见错误与诊断

| 症状 | 常见原因 | 检查 |
| --- | --- | --- |
| embedding index out of range | 新 token 未 resize 或 checkpoint 错 | 最大 ID 与 embedding 行数 |
| 输出乱码/异常切分 | tokenizer 与权重不匹配 | checkpoint、词表、special IDs |
| PAD 被当作内容 | 输入 mask 缺失或反了 | 打印 input_ids、attention_mask |
| PAD 目标仍贡献 loss | 未设置 ignore_index/label mask | 检查目标与 loss |
| 文本意外变短 | max_length/截断方向不合适 | 查看实际 token 序列 |
| 上下文溢出 | 按字符估算或漏算模板 | 用目标 tokenizer 计完整 prompt |
| decode 空格或标点不同 | decoder/normalization 有损 | 检查 tokens 和 offsets |

    sample = "检查这段输入。"
    encoded = tok(sample, return_tensors="pt")
    print(tok.convert_ids_to_tokens(encoded["input_ids"][0]))
    print(encoded["input_ids"][0].tolist())
    print(encoded["attention_mask"][0].tolist())
    print(tok.decode(encoded["input_ids"][0]))

## 22. 实践路线

1. 比较字符、词和子词切分长度与 OOV 行为。
2. 检查 special token 文本/ID 及 add_special_tokens 的差异。
3. 构造不同长度 batch，确认 PAD 对应 mask 为 0。
4. 用 offset mapping 完成 NER 子词标签对齐。
5. 使用聊天模型 chat template，测量模板后的完整 token 数。
6. 比较中英文和代码的字符/token 比。
7. 训练并保存小型 BPE，重载后验证编码一致。

学完后应能解释：文本如何切开、实际 token 数、特殊 token 是否正确、哪些位置是 PAD、模型是否认识这些 ID、截断丢掉了什么，以及序列长度如何影响上下文和成本。
