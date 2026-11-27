
import numpy as np
import openai

from core import model_helper

def test_embeddings():
    documents = ["退款需要几个工作日", "如何修改登录密码"]
    vectors = np.array(
        [
            [0.8, 0.1, 0.0],
            [0.0, 0.1, 0.9],
        ],
        dtype=np.float32,
    )
    query_vector = np.array([0.7, 0.2, 0.0], dtype=np.float32)

    assert vectors.shape == (len(documents), 3)
    assert query_vector.shape == (3,)





# ### 3.4 使用 OpenAI 兼容 Embedding API

def test_embeddings_2():
    embedding_model = model_helper.embedding_model
    with model_helper.get_openai_client() as client:
        response = client.embeddings.create(
            model=embedding_model,
            # `input` 可以是一条文本或一组文本。批量调用时要考虑单次请求的文本长度、Token 限制、服务商批量上限和失败重试。保存向量时还应记录 `embedding_model`、维度和索引版本。
            input=["退款需要几个工作日", "如何修改登录密码"],
        )

        vectors = [item.embedding for item in response.data]
        print(len(vectors))
        print(len(vectors[0]))
        print(response.model)
        print(response.usage)





