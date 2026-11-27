import uuid

import openai
from openai import OpenAI

print("OpenAI version:", openai.__version__)

from core import model_helper

from openai import AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError, BadRequestError,APIResponseValidationError

default_api_key = model_helper.qwen_api_key
default_base_url = model_helper.base_url
default_model = model_helper.model_chat
default_embedding_model = model_helper.embedding_model

def get_openai_client(api_key: str = default_api_key, base_url: str = default_base_url) -> OpenAI:
    return OpenAI(
        # 客户端请求持有的 Key
        api_key=api_key,
        # 请求将发送到哪个 API 根地址
        base_url=base_url,
        # 客户端级默认超时
        timeout=30.0,
        # 控制 SDK 层面的自动重试次数。
        max_retries=3,
    )

"""
`OpenAI` 对象不是“一个直接生成文本的方法”，而是一个同步 API 客户端和资源入口。真正的业务调用通常位于多级属性之后：

client.responses.create(...)
client.chat.completions.create(...)
client.embeddings.create(...)
client.models.list(...)
client.files.create(...)

"""

def test01_client_info():
    # 获取 client 里面的一些信息

    client = get_openai_client()
    print(f"client.base_url = {client.base_url}")
    print(f"client.api_key = {client.api_key}")
    print(f"client.timeout = {client.timeout}")
    print(f"client.max_retries = {client.max_retries}")
    # 确定当前客户端是否持有 api_key
    print(bool(client.api_key))

    # 当前支持的模型有哪些
    # 检查 Key 是否能访问模型列表。
    for model in client.models.list():
        print(f"client.model = {model}")


    model_info = client.models.retrieve(default_model)
    print(f"{default_model} 模型id：{model_info.id}")
    ...

def test02_invoke_path():
    """
    先理解请求链路：
    创建客户端
      -> 选择资源
      -> 调用资源方法
      -> 得到 SDK 类型化响应
      -> 读取响应属性
      -> 处理异常、超时、重试和资源关闭
    """
    client = get_openai_client()
    try:
        # do something
        print("client.responses.create(...)")
        print("client.chat.completions.create(...)")
        print("client.embeddings.create(...)")
        print("responses.stream(...)")
        print("parse response")
    except AuthenticationError:
        print("认证失败：检查 API Key 和项目权限。")
    except RateLimitError:
        print("请求受到限流：降低并发，稍后重试。")
    except BadRequestError as error:
        print(f"请求参数错误：{error}")
    except APITimeoutError:
        print("请求超时：检查网络或增大本次 timeout。")
    except APIConnectionError:
        print("连接失败：检查网络、代理和 base_url。")
    except APIResponseValidationError:
        print("服务端返回的数据不符合 SDK 预期结构。")
    except APIStatusError as error:
        print(
            "服务端错误",
            error.status_code,
            error.request_id,
            error.body,
        )
    else:
        print("No error")
    finally:
        print(f"closed 1 ? = {client.is_closed()}") # False
        print("Closing client")
        ### `client.is_closed()`、`client.close()` 和上下文管理器
        client.close()
        print(f"closed 2 ? = {client.is_closed()}") # True

    print("=================================")


    # 短生命周期脚本
    with get_openai_client() as client:
        print(f"closed 1 ? = {client.is_closed()}") # False
    print(f"closed 2 ? = {client.is_closed()}") # True

    ...



def test03_embedding():
    # ### 5.6 `client.embeddings`

    print("default_embedding_model: " , default_embedding_model)

    with get_openai_client() as client:
        response = client.embeddings.create(
            model=default_embedding_model.strip(),
            input=["what is token", "what is tokenizer"],
        )

        for item in response.data:
            print(item)
            print(item.embedding)

    ...


def test05_responses():
    ### 5.1 `client.responses`
# | `create()` | 创建一次模型响应 |
# | `parse()` | 创建响应并把结构化输出解析成指定 Python 类型 |
# | `stream()` | 以事件流方式读取响应 |
# | `retrieve()` | 根据响应 ID 查询已有响应 |
# | `cancel()` | 取消后台响应 |
# | `delete()` | 删除响应记录 |
# | `compact()` | 对上下文进行压缩，属于进阶上下文管理 |

    trace_id = uuid.uuid4().hex
    input = "what is token in large language model?"

    with get_openai_client() as client:
        # response 类所有可指定的参数
        # | `model` | 指定模型 |
        # | `input` | 文本、图片、文件或输入项 |
        # | `instructions` | 系统/开发者级指令 |
        # | `max_output_tokens` | 限制输出 Token 上限 |
        # | `temperature` | 控制随机性，是否支持取决于模型 |
        # | `tools` | 声明模型可使用的工具 |
        # | `tool_choice` | 控制工具选择策略 |
        # | `previous_response_id` | 关联前一响应，实现连续上下文 |
        # | `conversation` | 使用服务端会话资源 |
        # | `store` | 控制响应是否保存，具体行为看接口和服务策略 |
        # | `metadata` | 附加业务元数据，便于检索或观测 |
        # | `timeout` | 只覆盖本次请求的超时时间 |
        response = client.responses.create(
            model=default_model,
            input=input,
            # instructions 给这次请求下一个系统/开发者级指令
            instructions="你是一个简洁的中文学习助手。",
            extra_headers={"x-trace-id": trace_id},
            timeout=10.0,
        )

        # # 5.2 `response` 的必学属性
        print(f"response = {response}")
        print(f"response.output_text: {response.output_text}")
        print(f"response.usage: {response.usage}")
        print(f"response.id: {response.id}")
        print(f"response.status: {response.status}")
        print(f"response.model: {response.model}")  # 实际使用的模型
        print(f"response.error: {response.error}")  # 生成失败时的错误信息
        usage = response.usage
        if usage:
            print(usage.input_tokens)
            print(usage.output_tokens)
            print(usage.total_tokens)


    """
    ### 5.5 `responses` 和 `chat.completions` 如何选择

| 场景 | 建议 |
| --- | --- |
| 新项目、希望使用较新的平台能力 | 优先学习 `client.responses` |
| 第三方 OpenAI 兼容服务 | 先确认服务商支持情况，常见是 `client.chat.completions` |
| 已有大量 Chat Completions 代码 | 可以继续使用，再逐步评估迁移 |
| 需要多模态、工具、结构化输出 | 优先比较 `responses`，同时检查具体模型支持 |
| 只需要最小聊天 demo | 两者都可以，但要理解响应结构不同 |

不要把两种响应对象混用：

- Responses 使用 `response.output_text` 和 `response.output`。
- Chat Completions 使用 `completion.choices[0].message.content`。
    """

    ...


def test06_responses():

    trace_id = uuid.uuid4().hex
    input = "what is token in large language model?"

    with get_openai_client() as client:
        # response 类所有可指定的参数
        # | `model` | 指定模型 |
        # | `input` | 文本、图片、文件或输入项 |
        # | `instructions` | 系统/开发者级指令 |
        # | `max_output_tokens` | 限制输出 Token 上限 |
        # | `temperature` | 控制随机性，是否支持取决于模型 |
        # | `tools` | 声明模型可使用的工具 |
        # | `tool_choice` | 控制工具选择策略 |
        # | `previous_response_id` | 关联前一响应，实现连续上下文 |
        # | `conversation` | 使用服务端会话资源 |
        # | `store` | 控制响应是否保存，具体行为看接口和服务策略 |
        # | `metadata` | 附加业务元数据，便于检索或观测 |
        # | `timeout` | 只覆盖本次请求的超时时间 |

        with client.responses.stream(
                model=default_model,
                input=input,
                instructions="你是一个简洁的中文学习助手。",
                extra_headers={"x-trace-id": trace_id},
                timeout=10.0,
        ) as stream:
            for event in stream:
                # print("event:", event)
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)

                elif event.type == "response.completed":
                    print("\n[完成]")

            # - `stream`：事件迭代器，逐步接收事件。
            # - `final_response`：流读取完成后聚合出的最终响应。
            final_response = stream.get_final_response()
            print(final_response.usage)


from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    confidence: float

def test07_responses():
    """ 结构化输出 """

    trace_id = uuid.uuid4().hex

    with get_openai_client() as client:
        parsed_response = client.responses.parse(
            model=default_model,
            input="what is token in large language model?",
            instructions="你是一个简洁的中文学习助手。",
            extra_headers={"x-trace-id": trace_id},
            text_format=Answer,
            timeout=10.0,
        )
        print(parsed_response)
    ...


"""

错误处理 -> 常用异常层级

OpenAIError
  └── APIError
        ├── APIStatusError
        │     ├── BadRequestError       400
        │     ├── AuthenticationError  401
        │     ├── PermissionDeniedError 403
        │     ├── NotFoundError        404
        │     ├── RateLimitError       429
        │     └── InternalServerError  5xx
        ├── APIConnectionError
        │     └── APITimeoutError
        └── APIResponseValidationError
"""


