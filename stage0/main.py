import sys

import openai


def main():
    print("Hello from stage0!")

from core import model_helper
print(model_helper.qwen_api_key)

from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, ChatCompletion

def chat1():
    model = model_helper.model_chat
    # model = "gpt-5.4"

    client = OpenAI(
        api_key=model_helper.qwen_api_key,
        base_url=model_helper.base_url,
        timeout=30,
    )

    resp: ChatCompletion
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
    chat1()
    pass