from pathlib import Path

import loaders

# 文档切块


### 4.2 清洗原则
# 清洗不是“把所有符号删除”。标题、列表、代码、表格和段落结构可能是回答问题的重要信息。
import re


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def test_clean_text():
    cleaned = clean_text("标题\r\n\r\n\r\n内容\t  继续")
    print(cleaned)


### 4.3 从 Markdown 标题提取层级
# 保留最近的标题作为每个块的上下文：
def extract_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    headings: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            title = " / ".join(headings) if headings else "未命名章节"
            sections.append((title, content))

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            headings[:] = headings[: level - 1]
            headings.append(match.group(2))
            buffer.clear()
        else:
            buffer.append(line)
    flush()
    return sections


def test_extract_sections():

    text = """
    
# 神经网络语言模型

Bengio在2003年发表的论文: [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), 提出了基于神经网络的语言模型(NNLM).

![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/533/1670182687094/52b641a427f541b9b25b194833d7ab45.png)

本质上就是利用神经网络预测下一个词.

结构可以简化为: softmax（U[tanh(WC+b1)]+b2）

# 词向量（神经网络语言模型的副产品 C）

给我任何一个词

“hello” --》 独热编码w1 [1,0,0,0,0]

w1*C =c1 （“hello”这个词的词向量）

词向量：就是用一个向量来表示一个单词

可以控制词向量的维度（大小）

如果我们得到的词向量，第一个问题也被解决了，（下游任务）

# 总结

神经网络语言模型(NNLM)通过构建神经网络的方式来探索和建模自然语言内在的依赖关系。

## 优点：

  - 词向量是可以自定义维度的，维度并不会因为新扩展词而发生改变，词向量能够很好的根据特征距离度量词与词之间的相似性；
  - 好的词向量能够提高模型泛化能力；
  - 相比于n-gram，通过词向量的降维，减小了参数空间，减少了计算量
## 缺点:

  - 参数较多，模型训练时间长；
  - 神经网络黑盒子，可解释性较差。

    """

    res = extract_sections(text)
    print(len(res))
    print(res[0][0])
    print("="*90)
    print(res[0][1])
    ...

## 5. Chunk：文档切块

from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """ 定义文档块 """
    chunk_id: str
    text: str
    metadata: dict[str, object]



### 5.3 一个简单的重叠切块器

def split_by_chars(
    document: loaders.SourceDocument,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[DocumentChunk]:
    """
        切割原始文本为多一个 DocumentChunk文档块对象
    :param document: 原始文档对象
    :param chunk_size: 每一块的大小
    :param overlap: 每一块重叠的大小
    :return: 返回原始文档切割后的多个文档块
    """

    if chunk_size < 0:
        raise ValueError("chunk_size must be >= 0")
    if overlap < 0 or overlap > chunk_size:
        raise ValueError("overlap must be >= 0 and <= chunk_size")

    text = clean_text(document.text)
    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap # 每一步滑动窗口移动多少个位置

    for chunk_index, start in enumerate(range(0, len(text), step)):
        end = start + chunk_size
        if end > len(text):
            end = len(text) - 1
        piece = text[start:end].strip()
        if not piece:
            continue

        chunks.append(DocumentChunk(
            f"{document.source}::paragraph-{chunk_index}",
            piece,
            metadata={
                **document.metadata,
                "source": document.source,
                "title": document.title,
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": min(start + chunk_size, len(text)),
            },
        ))

    return chunks

# 注意这个算法可能在段落中间断开，也可能在最后形成很短的块。它用于理解 `chunk_size` 和 `overlap`，不是最终最优切分器。

def test_split_by_chars():

    file_path = "./data/神经网络语言模型.md"

    document = loaders.read_text_document(Path(file_path))

    res = split_by_chars(document, chunk_size=100, overlap=10)
    print(len(res))
    print(res[0])


    ...

### 5.4 先按段落，再按长度切分
# 更自然的第一版可以先把段落合并到不超过上限的块中；单个超长段落会先按字符上限拆开：

def split_by_paragraphs(
    document: loaders.SourceDocument,
    max_chars: int = 800,
    overlap_paragraphs: int = 1,
) -> list[DocumentChunk]:
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")
    if overlap_paragraphs < 0:
        raise ValueError("overlap_paragraphs 不能小于 0")

    paragraphs = [p.strip() for p in clean_text(document.text).split("\n\n") if p.strip()]
    bounded_paragraphs: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            bounded_paragraphs.append(paragraph)
        else:
            bounded_paragraphs.extend(
                paragraph[start : start + max_chars]
                for start in range(0, len(paragraph), max_chars)
            )
    paragraphs = bounded_paragraphs

    chunks: list[DocumentChunk] = []
    current: list[str] = []
    current_len = 0

    def emit(parts: list[str], index: int) -> DocumentChunk:
        text = "\n\n".join(parts)
        return DocumentChunk(
            chunk_id=f"{document.source}::paragraph-{index}",
            text=text,
            metadata={
                **document.metadata,
                "source": document.source,
                "title": document.title,
                "chunk_index": index,
            },
        )

    for paragraph in paragraphs:
        would_exceed = current and current_len + len(paragraph) + 2 > max_chars
        if would_exceed:
            chunks.append(emit(current, len(chunks)))
            current = current[-overlap_paragraphs:]
            current_len = sum(len(item) for item in current) + max(0, len(current) - 1) * 2
            if current and current_len + len(paragraph) + 2 > max_chars:
                current = []
                current_len = 0

        current.append(paragraph)
        current_len += len(paragraph) + (2 if len(current) > 1 else 0)

    if current:
        chunks.append(emit(current, len(chunks)))
    return chunks


import hashlib


def make_document_version(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def test_make_document_version():
    text = """

    # 神经网络语言模型

    Bengio在2003年发表的论文: [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), 提出了基于神经网络的语言模型(NNLM).

    ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/533/1670182687094/52b641a427f541b9b25b194833d7ab45.png)

    本质上就是利用神经网络预测下一个词.

    结构可以简化为: softmax（U[tanh(WC+b1)]+b2）

    # 词向量（神经网络语言模型的副产品 C）

    给我任何一个词

    “hello” --》 独热编码w1 [1,0,0,0,0]

    w1*C =c1 （“hello”这个词的词向量）

    词向量：就是用一个向量来表示一个单词

    可以控制词向量的维度（大小）

    如果我们得到的词向量，第一个问题也被解决了，（下游任务）

    # 总结

    神经网络语言模型(NNLM)通过构建神经网络的方式来探索和建模自然语言内在的依赖关系。

    ## 优点：

      - 词向量是可以自定义维度的，维度并不会因为新扩展词而发生改变，词向量能够很好的根据特征距离度量词与词之间的相似性；
      - 好的词向量能够提高模型泛化能力；
      - 相比于n-gram，通过词向量的降维，减小了参数空间，减少了计算量
    ## 缺点:

      - 参数较多，模型训练时间长；
      - 神经网络黑盒子，可解释性较差。

        """

    version = make_document_version(text)
    print(version)




