
import numpy as np
import openai

from core import model_helper
import retriever
import tiktoken

# Embedding 接口和批处理


### 3.5 封装 Embedding 客户端
class EmbeddingClient:
    def __init__(self, client: openai.OpenAI, model: str, batch_size: int = 32):
        """
            初始化嵌入模型客户端
        :param client: openai 官方嵌入模型客户端
        :param model: 向量模型名称
        :param batch_size: 嵌入向量批量查询大小
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """ 查询批量文本的嵌入向量 """
        result: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            end = min(start + self.batch_size, len(texts))
            batch = texts[start:end]
            response = self.client.embeddings.create(model=self.model, input=batch)
            items = sorted(response.data, key=lambda item: item.index)
            if len(items) != len(batch):
                raise ValueError("Batch size does not match")

            for item in items:
                result.append(item.embedding)

        return result

    def embed_query(self, text: str) -> list[float]:
        """ 查询单个文本的嵌入向量 """
        result: list[list[float]] = self.embed_documents([text])
        return result[0]


def test_embeddings_client():
    # 检查返回的 `index` 很重要：批量响应不应仅凭返回顺序猜测，尤其是换用异步/并行实现后。生产代码还应记录请求耗时、Token 用量和失败批次。

    embed_client = EmbeddingClient(model_helper.get_openai_client(), model_helper.embedding_model, batch_size=2)

    texts = ["命令行交互，用户输入一行", "命令行交互，用户输入多行且逐行处理", "循环一次性读取多行标准输入",
             "读取管道数据", "读取重定向文件", "编写大模型命令行交互程序"]

    vectors = embed_client.embed_documents(texts)

    vector = embed_client.embed_query(texts[0])

    scores = retriever.cosine_similarity(np.array(vector), np.array(vectors))

    print(f"scores: {scores}")



### 5.5 使用 Tokenizer 估算块长度
# 字符数不是 Token 数。可以用 `tiktoken` 对 OpenAI 模型做近似估计：

# tiktoken 只负责 Tokenizer（分词和编码），不负责生成 Embedding（嵌入向量）。
# Embedding 需要通过模型的 Embedding 层完成。

def text_tokens(text: str, embedding_model:str = "cl100k_base") -> list[int]:
    encoding = tiktoken.get_encoding(embedding_model)
    token_ids = encoding.encode(text)
    return token_ids

def num_text_tokens(text: str, embedding_model:str = "cl100k_base") -> int:
    return len(text_tokens(text, embedding_model))

def test_text_tokens():
    text = "这是一个用于估算 Token 数量的中文文本。"
    token_ids = text_tokens(text)
    print(token_ids)

    print(num_text_tokens(text, embedding_model="cl100k_base"))



# uv add torch transformers
import torch
from transformers import AutoTokenizer, AutoModel


# 模型	主要语言	适用场景
# bert-base-chinese	中文	中文 NLP
# bert-base-uncased	英文	英文 NLP，不区分大小写
# bert-base-cased	英文	英文 NLP，保留大小写
# bert-base-multilingual-cased	多语言	中英文混合、多语言 NLP

def token_id_embedding(text: str, embedding_model:str = "bert-base-chinese") -> torch.Tensor:
    # 1. 加载预训练 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    # 2. 加载预训练模型
    model = AutoModel.from_pretrained(embedding_model)

    # 3. 文本编码 pt表示 pytorch tensor
    inputs = tokenizer(text, return_tensors="pt")
    print(f"inputs: {inputs}")

    # 4. 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. 获取最后一层隐藏状态
    embedding = outputs.last_hidden_state

    print(embedding.shape)

    # 返回 嵌入
    return embedding

def test_token_id_embedding():
    text = "我正在学习大语言模型"

    token_ids = text_tokens(text)
    print(token_ids)

    embed = token_id_embedding(text)
    print(f"embed.shape: {embed.shape}")
