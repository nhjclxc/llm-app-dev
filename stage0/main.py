import json
import sys
import uuid
from pyexpat.errors import messages

import openai


def main():
    print("Hello from stage0!")

from core import model_helper
print(model_helper.qwen_api_key)

from openai import OpenAI, APIConnectionError, APIError, AuthenticationError


def get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=model_helper.qwen_api_key,
        base_url=model_helper.base_url,
        timeout=30,
        max_retries=3,
    )

def chat_response():
    model = model_helper.model_chat

    client = get_openai_client()
    print("client.api_key: ", client.api_key)
    print("client.api_key: ", bool(client.api_key))
    print("client.base_url: ", client.base_url)
    print("client.timeout: ", client.timeout)
    print("default_headers: ", client.default_headers)
    print("default_query: ", client.default_query)
    print("uuid.uuid4().hex: ", uuid.uuid4().hex)

    # if True:
    #     return
    try:
        trace_id =  uuid.uuid4().hex
        resp = client.responses.create(
            model=model,
            input="用两句话解释大语言模型应用开发是什么",
            extra_headers={"X-Trace-ID": trace_id},
            extra_query={"debug": "false"},
        )
        print(resp)
        print(resp.id)
        print(resp.model)
        print(resp.model_config)
        # print(resp.output)
        # for output in resp.output:
        #     if output.type is not None and  output.type == 'message':
        #         print(output.content[0].text)
        # print(resp.output[1])
        # print(resp.output[1].content)
        # print(resp.output[1].content[0].text)
        client.is_closed()
    finally:
        client.close()


    print(client.is_closed())


    pass

def chat_response_parse():
    client = get_openai_client()


    pass

def chat_completion_stream():
    client = get_openai_client()
    model = model_helper.model_chat
    resps = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是简洁、准确的中文学习助手。"},
            {"role": "user", "content": "用两句话解释大语言模型应用开发是什么。"},
        ],
        temperature=0.2,
        stream=True,
    )

    for resp in resps:
        if len(resp.choices) > 0 and resp.choices[0].delta.content is not None:
            print(resp.choices[0].delta.content, end="")
        pass
    pass

def chat_completion():

    model = model_helper.model_chat
    # model = "gpt-5.4"

    client = OpenAI(
        api_key=model_helper.qwen_api_key,
        base_url=model_helper.base_url,
        timeout=30,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是简洁、准确的中文学习助手。"},
                {"role": "user", "content": "用两句话解释大语言模型应用开发是什么。"},
            ],
            temperature=0.2,
        )
    except AuthenticationError:
        sys.exit("认证失败：请检查 API Key 是否正确、有效，或是否已经过期。")
    except APIConnectionError:
        sys.exit("连接失败：请检查网络、代理和 QWEN_BASE_URL 配置。")
    except APIError as error:
        sys.exit(f"模型服务返回错误：{error}")

    print(resp)
    print(resp.id)
    print(resp.choices[0])
    print(resp.choices[0].message)
    print(resp.choices[0].message.content)

    pass


if __name__ == "__main__":
    # main()
    # chat_response()
    chat_response_parse()
    # chat_completion()
    # chat_completion_stream()
    pass