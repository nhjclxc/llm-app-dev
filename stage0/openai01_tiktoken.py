import tiktoken


"""
How to count tokens with Tiktoken???
tiktoken is a fast open-source tokenizer by OpenAI.
通过给定文本获取对应的token_id

https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken

# 返回给定编码器
encoding = tiktoken.get_encoding(encoding_name)
# 返回给定模型对于的编码器
encoding = tiktoken.encoding_for_model(model_name)

"""

def test_tiktoken01():
    encoding = tiktoken.get_encoding("cl100k_base")
    assert encoding is not None


    # encode: 文本 -> id
    text = "tiktoken is great!"
    token_ids = encoding.encode(text)
    print(token_ids)

    print(f"len(token_ids) = {len(token_ids)}")
    print(f"len(token_ids) = {num_token_for_strings(text)}")
    text2 = "嵌入模型"
    print(f"token_ids = {token_for_strings(text2)}")

    # decode: id -> 文本
    decoded = encoding.decode(token_ids)
    print(f"decoded = {decoded}")

    # .decode_single_token_bytes()安全地将单个整数标记转换为它所表示的字节。
    res = [encoding.decode_single_token_bytes(token_id) for token_id in token_ids]
    print(f"res = {res}")


def token_for_strings(text: str, encoding_name: str = "cl100k_base") -> list[int]:
    encoding = tiktoken.get_encoding(encoding_name)
    assert encoding is not None
    token_ids = encoding.encode(text)
    return token_ids


def num_token_for_strings(text: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    return len(token_for_strings(text, encoding_name))

