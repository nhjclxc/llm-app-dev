import uuid
from pyexpat.errors import messages

import openai
from openai import OpenAI
from core import model_helper


default_api_key = model_helper.qwen_api_key
default_base_url = model_helper.base_url
default_model = model_helper.model_chat
default_embedding_model = model_helper.embedding_model

print("OpenAI version:", openai.__version__)


# 核心链路是：用户需求 -> 组织消息和 Prompt（控制上下文长度） -> 调用模型 API -> 读取响应/流 -> 展示结果（处理错误与安全边界）


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



def test01_completions():
    ## 2. 一个请求由什么组成
    ### 2.1 客户端、模型和消息

    trace_id = uuid.uuid4().hex

    with get_openai_client() as client:
        responses = client.chat.completions.create(
            model=default_model,
            messages = [
                {"role": "system", "content": "你是一个简洁准确的中文助手"},
                {"role": "user", "content": "What is token in large language model?"},
            ],
            extra_headers={"x-trace-id": trace_id},
        )

        choice = responses.choices[0]
        print(f"choice.message.content: {choice.message.content}")
        print(f"choice.finish_reason: {choice.finish_reason}")
        print(f"request_id: {responses.id}")
        print(f"model: {responses.model}")
        usage = responses.usage
        if usage is not None:
            print(f"输入 token: {usage.prompt_tokens}")
            print(f"输出 token: {usage.completion_tokens}")
            print(f"总计 token: {usage.total_tokens}")


def test02_completions():
    # 历史对话

    messages = [
        {"role": "system", "content": "你是一个python入门老师"},
        {"role": "user", "content": "什么是列表？列表和元组有什么区别？"},
    ]

    with get_openai_client() as client:
        responses = client.chat.completions.create(
            model=default_model,
            messages = messages,
        )
        choice = responses.choices[0]
        print(f"choice.message.content: {choice.message.content}")

    ...


def test03_completions():
    # 多轮对话

    # 将历史的聊天记录带回当前聊天中
    messages = [
        {"role": "system", "content": "你是 Python 入门老师，回答简洁。"},
        {"role": "user", "content": "什么是列表？"},
        {"role": "assistant", "content": "列表是有顺序、可修改的一组元素，例如 [10, 20]。"},
        {"role": "user", "content": "那它怎么添加元素？"},
    ]
    # 正确顺序是：追加用户消息 -> 发送当前历史 -> 读取助手回复 -> 将助手回复加入历史。仅把回复打印出来、却不加入历史，下一轮就会丢失上下文。

    with get_openai_client() as client:
        responses = client.chat.completions.create(
            model=default_model,
            messages=messages,
        )
        choice = responses.choices[0]
        print(f"choice.message.content: {choice.message.content}")

    ...

def test05_completions():
    # 历史消息的保留策略
    # ### 3.3 为什么历史不能无限增长
    # 模型有上下文窗口限制。一次请求的输入包括系统规则、历史、当前输入以及可能的工具/格式信息；输出也受模型和接口的限制。历史越长，通常成本和延迟越高，也更容易超过上限。
    # 常见策略：
    # 1. **保留最近若干轮**：实现简单，但可能删掉早期重要事实。
    # 2. **按 Token 预算截断**：比按字符数更贴近模型限制，但要使用对应模型的 Tokenizer，并预留输出空间。
    # 3. **摘要旧历史**：节省空间，但摘要可能漏掉或扭曲事实；应将摘要标记为压缩记忆，而不是原始对话。
    # 4. **混合策略**：保留系统规则、最近几轮和一份经过明确标记的旧对话摘要。


    ...


def trim_history(messages: list[dict[str, str]], max_turns: int) -> list[dict[str, str]]:
    # 第一条消息必须是 system，并且保留第一条消息。其余的就是保留最小的 max_turns
    if not messages or messages[0].get("role") != "system":
        raise ValueError("第一条消息必须是 'system'")
    system_message = messages[0]
    max_len = min(len(messages) -1, max_turns*2)
    print(f"max_len: {max_len}")
    return [system_message, messages[-max_len:]]


def test06_completions_stream():
    # chat completion 的 stream 流式输出

    with get_openai_client() as client:
        stream = client.chat.completions.create(
            model=default_model,
            messages=[
                {"role": "user", "content": "什么是token"}
            ],
            stream=True,
        )

        parts: list[str] = []

        for chunk in stream:
            # 流式响应中的 `delta.content` 可能为 `None`，例如事件只携带结束信息或其他元数据。因此不要假设每个 chunk 都有文本。
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                print(text, end="", flush=True)
                parts.append(text)
        print(f"len parts: {len(parts)}")
        print("\n---------------- end ----------------")

    ...



"""
OpenAI 客户端
├── client.chat.completions
│   ├── create()       普通生成或流式生成
│   ├── parse()        结构化输出并解析到 Python 类型
│   ├── stream()       Chat Completions 流式辅助上下文管理器
│   ├── retrieve/list/update/delete()  已保存补全的管理操作，依赖服务端支持
│   └── 返回 ChatCompletion / ChatCompletionChunk
└── client.responses
    ├── create()       普通生成或事件流生成
    ├── parse()        结构化输出并解析到 Python 类型
    ├── stream()       Responses 流式辅助上下文管理器
    ├── retrieve/cancel/delete()/compact() 响应生命周期与上下文操作
    └── 返回 Response / ResponseStreamEvent

"""


def test07_completions_tools():

    with get_openai_client() as client:
        stream = client.chat.completions.create(
            model=default_model,
            messages=[
                {"role": "system", "content": "现在你是一个中枢大脑, 我提供了天气工具 get_weather(date, city)"},
                {"role": "user", "content": "我现在想知道 北京 在2026年1月1号的天气"}
            ],
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            message = choice.delta

            if message.refusal:
                print("模型拒绝：", message.refusal)
            elif message.tool_calls:
                print("模型请求工具调用，尚未执行")
                for call in message.tool_calls:
                    print(call.id, call.function.name, call.function.arguments)
            elif message.content is not None:
                print(message.content, end="", flush=True)
            else:
                print("没有普通文本内容；检查 finish_reason 和完整响应")


















