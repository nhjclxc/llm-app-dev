# OpenAI SDK 文本调用详解：Chat Completions 与 Responses

> 适合对象：已完成 [阶段 1 学习文档](stage1.md) 中 API 基础部分，希望系统掌握 `client.chat.completions` 和 `client.responses` 的学习者。
>
> 版本基准：本文按本仓库 `stage1/uv.lock` 锁定的 `openai 3.59.0` 编写。SDK、模型和服务商接口持续变化；尤其是第三方 OpenAI 兼容服务，不能因为 SDK 暴露某个方法就假设服务端支持对应端点、参数或响应字段。
>
> 学习原则：先掌握同步文本调用、响应读取、流式输出和异常处理；再学习结构化输出与服务端响应管理。以下示例默认已经安全配置 `QWEN_API_KEY`、`QWEN_BASE_URL`、`QWEN_MODEL`，不会在代码中硬编码或打印密钥。

## 1. 先看对象层级

`OpenAI` 是客户端，资源对象代表 API 的一组操作，方法调用才会发起网络请求。`client.chat.completions` 和 `client.responses` 是两条不同的文本 API 路径，返回对象结构不同。

```text
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
```

一般文本调用常见的属性路径是：

```python
# Chat Completions
completion.choices[0].message.content

# Responses
response.output_text
```

不要把二者混写成一个路径。Responses 的 `output` 是多种输出项的集合；Chat Completions 的 `choices` 是候选回答集合。

## 2. 创建客户端与资源

### 2.1 从环境变量读取配置

仓库当前练习使用 OpenAI SDK 访问兼容服务。`QWEN_API_KEY` 不是 SDK 自动识别的默认变量，因此显式传给 `OpenAI`：

```python
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["QWEN_API_KEY"],
    base_url=os.environ["QWEN_BASE_URL"],
    timeout=30.0,
    max_retries=2,
)
model = os.environ["QWEN_MODEL"]
```

- `api_key`：认证凭证。不要打印、提交或放进前端代码。
- `base_url`：API 根地址，不是完整的 `/chat/completions` 路径。
- `timeout`：客户端默认请求超时；单次调用可覆盖。
- `max_retries`：SDK 对部分可重试错误的重试次数，不是无限重试，也不适用于所有错误。

短命令行程序可使用上下文管理器关闭连接：

```python
with OpenAI(
    api_key=os.environ["QWEN_API_KEY"],
    base_url=os.environ["QWEN_BASE_URL"],
) as client:
    result = client.chat.completions.create(
        model=os.environ["QWEN_MODEL"],
        messages=[{"role": "user", "content": "你好"}],
    )
    print(result.choices[0].message.content)
```

长生命周期服务通常在应用启动时创建共享客户端，在应用关闭时关闭，不要每个 HTTP 请求都创建新客户端。

### 2.2 资源对象不是响应对象

```python
resource = client.chat.completions  # API 资源入口，本身不代表一次模型回答
completion = resource.create(...)   # 发起网络请求并得到响应对象

responses_resource = client.responses
response = responses_resource.create(...)
```

SDK 属性链便于定位接口，网络 I/O 发生在调用方法时。客户端构造成功只说明本地配置通过初步检查，不证明 Key 有效、模型可用或网络连通。

## 3. Chat Completions

### 3.1 `create()`：最小非流式调用

`create()` 接收模型名和消息列表，返回 `ChatCompletion`：

```python
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "你是简洁、准确的中文助手。"},
        {"role": "user", "content": "用两句话解释什么是 Token。"},
    ],
)

answer = completion.choices[0].message.content
print(answer)
```

此处 `system` 设定任务规则，`user` 提供当前输入。多轮对话需要应用在下一次请求中再次提交历史消息；服务端不会仅凭同一个 `client` 对象自动记住本地聊天。

### 3.2 常用 `create()` 参数

`openai 3.59.0` 的签名包含大量参数，下面按用途分类。参数是否被目标模型和服务商接受，仍需查对应文档。

| 参数 | 常见用途 | 使用提醒 |
| --- | --- | --- |
| `model` | 选择模型 | 模型名称由服务商决定；可用模型不代表具备所有能力 |
| `messages` | system/developer/user/assistant/tool 消息历史 | 内容可能是文本或多模态结构，须符合接口约定 |
| `temperature` | 控制采样随机性 | 并非所有模型都支持；低温不保证事实正确 |
| `top_p` | 核采样范围 | 通常不要与 temperature 同时无目的地调整 |
| `max_completion_tokens` | 限制生成 Token 数 | 新代码优先看此参数；模型支持情况不同 |
| `max_tokens` | 旧式输出上限参数 | 有些模型已弃用或不支持，优先使用服务文档推荐参数 |
| `n` | 生成候选条数 | 多个候选会增加费用；读取时不能永远假设只有一个 choice |
| `stop` | 配置停止序列 | 可能让回答提前终止；不同模型行为不同 |
| `presence_penalty` | 调整新主题出现倾向 | 生成控制参数，不是事实校验 |
| `frequency_penalty` | 调整重复词的倾向 | 不适合用来代替 Prompt 中的清晰约束 |
| `tools` | 声明模型可请求的工具 | 模型只提出调用；应用负责鉴权、校验和执行 |
| `tool_choice` | 控制是否/如何选择工具 | `auto` 不等于程序自动安全执行 |
| `parallel_tool_calls` | 是否允许并行工具调用 | 需实现并发安全和每个调用的参数校验 |
| `response_format` | 指定 JSON/JSON Schema 等格式 | 支持情况依模型；严格格式仍要做业务校验 |
| `stream` | 逐片返回 | 返回类型变成流，不再是完整 `ChatCompletion` |
| `stream_options` | 流式选项，例如请求用量 | 是否提供最终 usage 依端点能力 |
| `metadata` | 为已保存对象附加元数据 | 不应放秘密或大量用户内容 |
| `store` | 控制服务端存储行为 | 数据留存策略和默认值要查服务商政策 |
| `user` / `safety_identifier` | 服务端安全或滥用追踪标识 | 不要传入可识别个人身份的明文信息，按平台说明使用 |
| `extra_headers` | 单请求附加 HTTP Header | 常用于 trace ID；不要覆盖认证头 |
| `extra_query` | 单请求附加查询参数 | 只在服务商有明确要求时使用 |
| `extra_body` | 附加 SDK 标准参数未覆盖的 JSON 字段 | 绑定服务商扩展，降低可移植性 |
| `timeout` | 覆盖单请求超时 | 不能替代整体请求预算和用户侧取消 |

简单任务示例：

```python
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "只根据材料总结，最多输出三条。"},
        {"role": "user", "content": "材料：团队完成登录页，修复了搜索错误。"},
    ],
    temperature=0.2,
    max_completion_tokens=250,
    timeout=20.0,
)
```

不要为了“显式”而把所有可选参数都填上。先使用最少参数让任务跑通，再针对可复现的问题调整单个参数。

### 3.3 `ChatCompletion` 顶层属性

常用顶层属性如下：

| 属性 | 含义 | 示例 |
| --- | --- | --- |
| `id` | 本次补全的 ID | `completion.id` |
| `object` | 对象类型标识 | 通常为 `chat.completion` |
| `created` | 创建时间的 Unix 时间戳 | `completion.created` |
| `model` | 服务实际返回的模型标识 | `completion.model` |
| `choices` | 一个或多个候选结果 | `completion.choices` |
| `usage` | 输入、输出及总用量；可能为空或不提供 | `completion.usage` |
| `system_fingerprint` | 服务端配置/系统指纹，若服务提供 | `completion.system_fingerprint` |
| `service_tier` | 服务等级信息，若返回 | `completion.service_tier` |

读取示例：

```python
print("id:", completion.id)
print("object:", completion.object)
print("created:", completion.created)
print("model:", completion.model)
print("choices:", len(completion.choices))
print("usage:", completion.usage)
```

响应对象基于 SDK 的类型化模型，可使用 `model_dump()` 转成 Python 字典用于受控序列化：

```python
payload = completion.model_dump()
print(payload.keys())
```

完整请求/响应可能含用户隐私、工具参数或其他敏感数据，不要把 `model_dump()` 无筛选地写入公共日志。

### 3.4 `choices`、`Choice` 和 `message`

`choices` 是候选结果列表。通常只请求一个结果时读 `[0]`；设置 `n > 1` 时应遍历。

```python
for choice in completion.choices:
    print("index:", choice.index)
    print("finish_reason:", choice.finish_reason)
    print("message role:", choice.message.role)
    print("message content:", choice.message.content)
```

`Choice` 常用属性：

- `index`：候选在 `choices` 中的索引。
- `message`：助手消息对象。
- `finish_reason`：该候选为什么结束，例如正常停止、长度上限或工具调用。具体枚举依服务端协议。
- `logprobs`：请求且模型支持时返回的 Token 概率信息。

`message` 常见属性：

- `role`：通常是 `assistant`。
- `content`：文本内容；工具调用时可能为 `None`。
- `refusal`：模型拒绝相关内容，是否出现依模型和响应格式。
- `tool_calls`：模型请求调用的工具列表。
- `function_call`：旧式函数调用字段，较新的实现应优先学习 `tool_calls`。
- `audio`、`annotations` 等：音频或特定功能可能出现的字段。

安全读取：

```python
if not completion.choices:
    raise RuntimeError("模型没有返回 choices")

choice = completion.choices[0]
message = choice.message

if message.refusal:
    print("模型拒绝：", message.refusal)
elif message.tool_calls:
    print("模型请求工具调用，尚未执行")
    for call in message.tool_calls:
        print(call.id, call.function.name, call.function.arguments)
elif message.content is not None:
    print(message.content)
else:
    print("没有普通文本内容；检查 finish_reason 和完整响应")
```

不要把 `finish_reason == "length"` 的截断文本当作完整结果。工具调用场景中，模型返回的 arguments 是不可信输入，执行前必须解析、校验权限和参数。

### 3.5 `usage` 属性

Chat Completions 的常见计量字段：

```python
usage = completion.usage
if usage is not None:
    print("prompt:", usage.prompt_tokens)
    print("completion:", usage.completion_tokens)
    print("total:", usage.total_tokens)
    print("prompt detail:", usage.prompt_tokens_details)
    print("completion detail:", usage.completion_tokens_details)
```

- `prompt_tokens`：输入消息所消耗的 Token 数。
- `completion_tokens`：生成内容所消耗的 Token 数。
- `total_tokens`：总数，通常是前两者合计。
- `prompt_tokens_details`、`completion_tokens_details`：模型支持时的细分用量，例如缓存或推理相关计量。

流式调用默认不一定在每个 chunk 中附带最终用量。部分兼容服务支持通过 `stream_options` 请求最后一个 chunk 提供 usage；先查服务商文档，不要假设可用。

### 3.6 多轮消息历史

Chat Completions 使用调用方提供的 `messages` 表示上下文。SDK 不会自动替你保存本地聊天状态：

```python
messages = [
    {"role": "system", "content": "你是 Python 入门老师，回答简洁。"},
    {"role": "user", "content": "什么是列表？"},
]

first = client.chat.completions.create(model=model, messages=messages)
answer = first.choices[0].message.content or ""
messages.append({"role": "assistant", "content": answer})

messages.append({"role": "user", "content": "如何添加元素？"})
second = client.chat.completions.create(model=model, messages=messages)
print(second.choices[0].message.content)
```

应用应决定如何持久化、隔离用户会话以及控制 Token 预算。系统消息、用户消息和被引用的外部文本具有不同信任级别；不要把用户内容拼成高优先级指令。

### 3.7 `parse()`：把输出解析为 Pydantic 类型

`parse()` 与 `create()` 相似，但可通过 `response_format` 指定 Pydantic 模型，并在返回消息上读取 `parsed`：

```python
from typing import Literal

from pydantic import BaseModel, Field


class TextAnalysis(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    summary: str = Field(max_length=80)
    urgent: bool


parsed_completion = client.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": "分析用户反馈并返回指定结构。"},
        {"role": "user", "content": "付款后订单一直显示处理中。"},
    ],
    response_format=TextAnalysis,
)

message = parsed_completion.choices[0].message
if message.parsed is not None:
    analysis: TextAnalysis = message.parsed
    print(analysis.category, analysis.summary, analysis.urgent)
elif message.refusal:
    print("模型拒绝：", message.refusal)
else:
    print("没有可解析结果，检查 finish_reason 和服务商支持情况")
```

注意：

- `response_format` 接受 Pydantic 类型时，SDK 会协助构造受约束格式并解析结果。
- `parsed` 为空时要处理拒答、截断或服务端不支持等情况。
- 结构化输出不是业务授权。仍需检查枚举、范围、关联 ID 和业务不变量。
- 第三方兼容服务可能不支持 `parse()` 使用的 JSON Schema 能力。

### 3.8 `stream=True`：直接迭代 Chat Completions 分片

在 `create()` 中传 `stream=True`，返回 `Stream[ChatCompletionChunk]`，而不是常规的 `ChatCompletion`：

```python
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "用三句话解释上下文窗口。"}],
    stream=True,
)

parts: list[str] = []
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
        parts.append(delta.content)
print()
answer = "".join(parts)
```

`ChatCompletionChunk` 常见顶层属性包括 `id`、`object`、`created`、`model`、`choices` 和可选的 usage。每个 choice 的 `delta` 表示本次新增内容，而不是完整 message：

```python
chunk = next(iter(stream))
if chunk.choices:
    delta = chunk.choices[0].delta
    print(delta.role)       # 通常只在开头分片出现
    print(delta.content)    # 文本增量，可能为 None
    print(delta.tool_calls) # 工具调用也可能分片到达
```

流可能以无文本的结束 chunk 收尾，也可能在已输出部分内容后遇到网络错误。只有确认正常结束后才应将拼接文本作为完整回答保存；中途失败时提示用户这是未完成内容。

### 3.9 `stream()`：流式上下文管理器

SDK 3.59.0 也提供 `client.chat.completions.stream()` 辅助 API。它包装底层流，并提供事件、累积结果和上下文管理：

```python
with client.chat.completions.stream(
    model=model,
    messages=[{"role": "user", "content": "解释一下流式输出。"}],
) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)

    final_completion = stream.get_final_completion()
    print("\n结束原因:", final_completion.choices[0].finish_reason)
```

和 `create(stream=True)` 相比：

- `create(stream=True)` 直接暴露 chunk，适合清楚理解协议及自行累积。
- `stream()` 提供更高层的事件和最终累积对象。
- 两者都是同步接口；异步应用应使用相应的 Async 客户端能力。

事件名称及对象以 SDK 版本为准。此接口支持的事件通常包括文本增量、消息完成、工具调用参数增量等；不要将未完成的工具参数分片直接执行。

### 3.10 Chat Completions 资源上的管理型方法

在 SDK 3.59.0 中资源还暴露这些方法：

| 方法 | 功能 | 重要限制 |
| --- | --- | --- |
| `retrieve(completion_id)` | 按 ID 读取已保存的 Chat Completion | 对象必须可检索且服务端支持；普通调用不一定会保存 |
| `list(...)` | 分页列出已保存的补全，可按模型、元数据等过滤 | 并非每个兼容服务都实现；注意分页和数据权限 |
| `update(completion_id, metadata=...)` | 更新已保存补全的元数据 | 不是修改已经生成的回答 |
| `delete(completion_id)` | 删除已保存的补全记录 | 依赖服务端存储功能和权限 |
| `messages` | 与已保存补全消息相关的子资源 | 属于较进阶的持久化接口，查看当前 SDK 与服务文档 |

仅在确认服务支持存储/管理时使用：

```python
# 这些示例需要服务端支持相应资源；不要把它们当成所有兼容接口的通用能力。
saved = client.chat.completions.retrieve("completion-id")
print(saved.id, saved.model)

page = client.chat.completions.list(limit=20, order="desc")
for item in page:
    print(item.id, item.model)

client.chat.completions.update(
    "completion-id",
    metadata={"lesson": "stage1"},
)

client.chat.completions.delete("completion-id")
```

这些方法与“在 Python 列表中维护多轮聊天历史”不是一回事。初学阶段的对话记忆通常由应用自己持久化；服务端对象管理是另一种能力，需评估留存、隐私和删除策略。

## 4. Responses API

### 4.1 `create()`：最小调用

Responses 接收 `input`，可另外提供 `instructions`。文本任务可以这样调用：

```python
response = client.responses.create(
    model=model,
    instructions="你是简洁、准确的中文助手。",
    input="用两句话解释什么是 Token。",
    max_output_tokens=300,
)

print(response.output_text)
```

`input` 可以是简单字符串，也可以是带角色、内容或多模态项目的结构化输入。复杂输入格式要参考 Responses 文档；不要假设任意 Chat Completions 消息结构都可原样传入。

### 4.2 常用 `create()` 参数

| 参数 | 常见用途 | 提醒 |
| --- | --- | --- |
| `model` | 选择模型 | 能力、名称和接口可用性依服务商 |
| `input` | 用户输入或输入项目列表 | 可包含文本及模型支持的其他模态 |
| `instructions` | 高优先级任务说明 | 新请求调用时显式传入，或使用提示模板；续接时注意规则是否仍需发送 |
| `max_output_tokens` | 限制输出总 Token 上限 | reasoning 模型可能把推理 Token 计入总上限，输出可少于预期 |
| `temperature` / `top_p` | 采样控制 | 某些模型不支持或忽略；不要无目的同时调 |
| `text` | 文本输出配置、格式或 verbosity | 可配置结构化输出；细节依模型和模式 |
| `tools` / `tool_choice` | 声明并选择工具 | 模型返回调用请求，程序仍须执行和回传结果 |
| `max_tool_calls` | 限制工具调用数 | 仍建议应用层有循环/预算上限 |
| `parallel_tool_calls` | 是否并行调用工具 | 应用必须保证并发执行安全 |
| `previous_response_id` | 将本次响应接在前一响应之后 | 服务端需存储并允许检索关联上下文 |
| `conversation` | 关联服务端会话 | 会话可用性、持久化和租户隔离依服务端 |
| `store` | 控制响应是否存储 | 结合服务商留存政策理解，不能只根据参数名推断数据政策 |
| `background` | 创建后台响应 | 要配合状态轮询、取消和结果读取，属于进阶功能 |
| `truncation` | 控制上下文超限处理策略 | 自动截断可能丢弃较早内容；业务关键内容应自行管理 |
| `reasoning` | 支持模型的推理配置 | 模型专属参数，不能假设所有模型都支持 |
| `include` | 请求附加响应数据 | 只在需要相应数据且服务支持时使用 |
| `metadata` | 附加少量业务元数据 | 不放密钥、个人敏感信息或大段文本 |
| `prompt` | 使用托管 Prompt/模板标识及变量 | 依平台能力，需管理模板版本 |
| `stream` / `stream_options` | 流式响应及相关选项 | 返回类型改变为事件流 |
| `extra_headers` / `extra_query` / `extra_body` | 服务商或链路扩展 | 可能降低可移植性，谨慎使用 |
| `timeout` | 覆盖单次请求超时 | 长任务也要有总时限与取消策略 |

典型文本任务：

```python
response = client.responses.create(
    model=model,
    instructions=(
        "你是严谨的中文编辑。仅根据材料总结，不补充材料外的事实。"
    ),
    input="材料：团队本周完成登录页，并修复了搜索错误。总结为两条。",
    max_output_tokens=250,
    temperature=0.2,
    timeout=20.0,
)
```

接口、模型和服务端对参数的组合有约束。例如某些推理模型不接受 `temperature`；若报参数错误，先查看模型文档，再用最小请求逐个添加可选项。

### 4.3 `Response` 顶层属性

Responses 返回的 `Response` 对象常用属性：

| 属性 | 含义 | 说明 |
| --- | --- | --- |
| `id` | 响应 ID | 用于追踪或后续 retrieve/cancel；需服务端保留 |
| `object` | 对象类型标识 | 常见为 `response` |
| `created_at` | 创建时间 | 通常是 Unix 时间戳 |
| `model` | 实际模型标识 | 以服务端返回值为准 |
| `status` | 响应状态 | 可能为 `completed`、`failed`、`incomplete` 等 |
| `output` | 完整输出项目列表 | 可包含消息、工具调用、拒绝等不同类型 |
| `output_text` | SDK 聚合出的文本便捷属性 | 适合一般文本任务；复杂多模态/工具流程要检查 `output` |
| `usage` | Token 用量对象 | 完成后可能包含 input/output/total 和细分字段 |
| `error` | 失败信息 | 状态失败时检查；也要处理异常抛出情况 |
| `incomplete_details` | 不完整原因 | 例如达到输出上限，字段依响应而定 |
| `previous_response_id` | 上一响应关联 ID | 若请求使用了响应续接，检查此关联 |
| `conversation` | 会话关联信息 | 服务端支持时返回 |
| `instructions` / `metadata` | 指令或元数据（若响应提供） | 不要将其视为安全权限系统 |
| `service_tier`、`temperature` 等 | 请求或服务信息 | 是否出现取决于接口版本和服务端 |

基础读取：

```python
print("id:", response.id)
print("status:", response.status)
print("model:", response.model)
print("text:", response.output_text)
print("usage:", response.usage)
print("error:", response.error)
print("incomplete:", response.incomplete_details)
```

`output_text` 是 SDK 提供的便捷聚合结果，不代表 `output` 中只有一个文本块。对于工具、拒答或多模态流程，要按 `item.type` 分支读取。

### 4.4 遍历 `output` 与内容块

Response 的 `output` 包含类型化项目。常见的文本消息形状为 `message -> content -> output_text`：

```python
for item in response.output:
    print("item type:", item.type)

    if item.type == "message":
        for content in item.content:
            if content.type == "output_text":
                print(content.text)
            elif content.type == "refusal":
                print("拒绝内容:", content.refusal)
    elif item.type == "function_call":
        print("工具调用:", item.name, item.arguments)
```

不同任务会有不同 output item，例如函数调用、内置工具结果或其他模态。不要假定 `response.output[0].content[0].text` 永远存在：第一项可能不是消息，消息也可能包含多个内容块。

若仅需普通文本，用：

```python
text = response.output_text
```

若需要准确判断状态、拒绝、工具调用或引用，遍历 `response.output` 并处理未识别类型。

### 4.5 Responses 的用量

常见用法：

```python
usage = response.usage
if usage is not None:
    print("input:", usage.input_tokens)
    print("output:", usage.output_tokens)
    print("total:", usage.total_tokens)
    print("input details:", usage.input_tokens_details)
    print("output details:", usage.output_tokens_details)
```

具体细分字段会随模型能力而变化。用量用于观测和估算成本，不应硬编码成唯一计费真相；费用要以服务商账单规则为准。

### 4.6 `parse()`：Responses 结构化输出

`responses.parse()` 接收 `text_format`，并将结构化结果放在 `output_parsed`：

```python
from typing import Literal

from pydantic import BaseModel, Field


class SummaryResult(BaseModel):
    language: Literal["zh", "en"]
    bullets: list[str] = Field(max_length=5)


parsed_response = client.responses.parse(
    model=model,
    instructions="总结材料，不补充原文外事实。",
    input="团队完成登录页，并修复了搜索错误。",
    text_format=SummaryResult,
)

result: SummaryResult | None = parsed_response.output_parsed
if result is not None:
    print(result.language)
    for bullet in result.bullets:
        print("-", bullet)
else:
    print("没有解析到结构化结果；检查拒绝、状态和服务端支持。")
```

Responses 还可以用 `text` 参数配置 JSON Schema 等格式；当希望 SDK 解析成 Python 类型时，`parse(text_format=...)` 更直接。即使解析成功，仍需执行业务规则校验。

### 4.7 多轮续接：`previous_response_id`

Responses 可通过前一个响应 ID 让服务端延续上下文：

```python
first = client.responses.create(
    model=model,
    instructions="你是 Python 入门老师。",
    input="什么是列表？",
)

second = client.responses.create(
    model=model,
    previous_response_id=first.id,
    input="如何添加元素？",
)
print(second.output_text)
```

需要理解：

- 这是服务端响应关联，不是 SDK 自动保存任意本地会话。
- 服务端必须支持并保留响应；若 `store=False` 或服务商不提供存储，续接可能不可用。
- 后续调用的 `instructions` 处理方式与普通输入历史不同。若关键规则必须每轮生效，应按接口文档在每次请求显式提供，并验证服务端的优先级规则。
- 响应 ID 不应被当作授权凭证；仍需在应用数据库中验证当前用户是否有权使用该会话。

若需要完全由应用控制，可将历史整理成 `input` 并自行存储，但应预算 Token 并保留消息角色和信任边界。

### 4.8 `stream=True`：原始 Responses 事件流

`responses.create(stream=True)` 返回事件迭代器，而非普通 `Response`：

```python
events = client.responses.create(
    model=model,
    instructions="你是简洁的中文助手。",
    input="用三句话介绍 Prompt。",
    stream=True,
)

for event in events:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.completed":
        print("\n[响应完成]", event.response.id)
    elif event.type == "response.failed":
        print("\n[响应失败]", event.response.error)
    elif event.type == "response.incomplete":
        print("\n[响应不完整]", event.response.incomplete_details)
```

Responses 流式以带 `type` 的事件为主，文本 delta 事件中读取 `delta`。可能遇到的事件包括响应创建、文本增量、文本结束、输出项开始/结束、响应完成/失败/不完整等。不要只处理文本 delta 而忽略最终状态。

### 4.9 `stream()`：Responses 流式上下文管理器

SDK 的辅助流 API 会聚合事件，并提供最终响应：

```python
with client.responses.stream(
    model=model,
    instructions="你是简洁的中文助手。",
    input="什么是上下文窗口？",
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)

    final_response = stream.get_final_response()

print("\nstatus:", final_response.status)
print("usage:", final_response.usage)
```

`with` 确保流资源在结束时释放；`get_final_response()` 应在事件迭代完成后读取。流中断时可能拿不到完整最终对象，应用需要提示“生成中断”，不能误报成功。

### 4.10 Responses 管理及上下文方法

SDK 3.59.0 中，`client.responses` 还提供以下方法：

| 方法 | 用途 | 边界 |
| --- | --- | --- |
| `retrieve(response_id, ...)` | 读取已保存响应，可选择 include 内容 | 需要响应存储和服务端支持 |
| `cancel(response_id)` | 取消后台运行中的响应 | 通常用于 background response；已完成响应不能靠它撤回 |
| `delete(response_id)` | 删除服务端响应记录 | 删除范围、留存和级联行为依平台政策 |
| `compact(...)` | 将输入/上下文压缩为适合继续交互的形式 | 属于上下文管理能力，模型与服务支持有要求 |
| `input_items.list(response_id, ...)` | 列出某个响应关联的输入项目 | 检查分页和服务兼容性 |
| `input_tokens.count(...)` | 估算输入 Token | 不是最终计费结果；模型和参数支持需确认 |

常见管理调用示意（先确认服务端支持）：

```python
saved = client.responses.retrieve(response.id)
print(saved.id, saved.status, saved.output_text)

# 后台响应才通常需要查询/取消生命周期
cancelled = client.responses.cancel("background-response-id")
print(cancelled.status)

# 删除服务端保存记录会改变远端状态，先确认数据保留和权限要求
deleted = client.responses.delete("response-id")
print(deleted.id, deleted.deleted)
```

`compact()` 和 Token 计数较适合作为后续专题。压缩可能省上下文，但不保证保留所有细节；关键事实应由应用保存并在压缩后验证。

## 5. 两种 API 的核心差别

| 维度 | Chat Completions | Responses |
| --- | --- | --- |
| 请求主要字段 | `messages` | `input`，可另给 `instructions` |
| 普通文本读取 | `choices[0].message.content` | `output_text` |
| 完整结果结构 | `choices -> message` | `output` 中的类型化项目 |
| 流式形态 | `ChatCompletionChunk` 的 `choices[].delta` | 多种事件，按 `event.type` 分发 |
| 结构化解析 | `parse(response_format=...)` -> `choices[].message.parsed` | `parse(text_format=...)` -> `output_parsed` |
| 多轮上下文 | 调用方重发 `messages` 历史 | 可传 `previous_response_id`，或自行管理 input 历史 |
| 兼容性 | 常见第三方兼容端点 | 服务商可能尚未实现 |
| 工具表示 | `message.tool_calls` | `output` 中的 function/tool call 项 |

选择建议：

- 已有 Chat Completions 程序或服务商只实现它：继续使用并做好适配。
- 新项目面向官方平台且需要 Responses 特有能力：优先评估 Responses。
- 希望跨供应商迁移：将 SDK 响应封装到自己的业务接口，不要让业务代码到处依赖某一家响应字段。
- 需要多模态、工具或结构化输出：以目标模型和服务商当前支持矩阵为准，不以 API 名称推断能力。

## 6. 通用异常处理与诊断

```python
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

try:
    response = client.responses.create(
        model=model,
        input="解释 API。",
        timeout=30.0,
    )
except AuthenticationError:
    print("认证失败：检查 Key、服务商和权限。")
except RateLimitError:
    print("请求限流：检查额度、并发和重试策略。")
except APITimeoutError:
    print("请求超时：检查网络并设置合理超时。")
except APIConnectionError:
    print("连接失败：检查 DNS、代理和 Base URL。")
except BadRequestError as error:
    print("请求不合法：检查模型、字段和参数组合。")
except APIStatusError as error:
    print("服务端错误状态:", error.status_code)
    print("请求 ID:", error.request_id)
```

- `AuthenticationError`：凭证无效或认证方式不匹配。
- `RateLimitError`：请求频率、并发或额度受到限制。
- `APITimeoutError`：请求等待超时。
- `APIConnectionError`：连接建立或传输失败。
- `BadRequestError`：参数、模型或输入不符合要求。
- `APIStatusError`：其他 HTTP 错误，可查看状态码和请求 ID。

记录错误类型、请求 ID、模型、耗时等诊断信息；不要记录 API Key，也不要默认记录完整 Prompt、响应或个人数据。重试要有限并区分临时故障与确定性错误。

## 7. 兼容服务与能力探测

SDK 资源方法只是客户端代码能力，不代表远端服务器一定实现。典型差异包括：

- Chat Completions 可用，但 `/responses` 返回 404。
- `response_format`、`stream_options`、`max_completion_tokens` 等字段不支持。
- 流式协议与 OpenAI 官方事件不同。
- 不提供模型列表、响应存储、retrieve/list/delete 等管理端点。
- `usage`、`finish_reason` 或错误体字段缺失/不同。
- 模型名称、参数范围和可用能力由服务商单独定义。

集成新服务商时按此顺序验证：

1. 阅读服务商的兼容性矩阵，确认端点、模型和参数。
2. 用最小非流式请求测试认证和响应读取。
3. 单独测试流式、结构化输出和上下文续接，不要默认它们一起支持。
4. 将响应 ID、用量和错误情况与真实返回核对。
5. 对不支持的能力在应用启动或功能入口处明确降级，避免用户只看到模糊 500 错误。

一个不调用模型的客户端层检查只能确认配置已载入：

```python
print("key loaded:", bool(client.api_key))
print("base url:", client.base_url)
print("model configured:", bool(model))
```

不要通过打印 `client.api_key` 来调试。

## 8. 实用封装：统一读取文本

不同 API 的结果结构不同，可以在应用边界统一成自己的返回对象。简单文本场景示例：

```python
from dataclasses import dataclass


@dataclass
class ModelTextResult:
    text: str
    request_id: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None


def call_chat(client, model: str, messages: list[dict[str, str]]) -> ModelTextResult:
    result = client.chat.completions.create(model=model, messages=messages)
    if not result.choices:
        raise RuntimeError("Chat Completions 没有返回候选结果")

    choice = result.choices[0]
    if choice.message.content is None:
        raise RuntimeError("本次结果不是普通文本；可能是拒答或工具调用")

    usage = result.usage
    return ModelTextResult(
        text=choice.message.content,
        request_id=result.id,
        model=result.model,
        input_tokens=usage.prompt_tokens if usage else None,
        output_tokens=usage.completion_tokens if usage else None,
    )


def call_responses(client, model: str, prompt: str) -> ModelTextResult:
    result = client.responses.create(model=model, input=prompt)
    if result.status != "completed":
        raise RuntimeError(
            f"Responses 未正常完成：status={result.status}, "
            f"incomplete={result.incomplete_details}"
        )

    usage = result.usage
    return ModelTextResult(
        text=result.output_text,
        request_id=result.id,
        model=result.model,
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
    )
```

此封装只适用于“单段文本回答”。遇到工具调用、拒答、图像/音频输出、多个候选或部分完成状态时，应设计更丰富的结果类型，而不是丢掉信息。

## 9. 常见误区

### 9.1 把 API 资源名当作方法

错误理解：`client.responses` 已经完成了生成。正确理解：它是资源入口，需要调用 `create()`、`parse()` 等方法。

### 9.2 把两种响应属性混用

```python
# Chat Completions
text = completion.choices[0].message.content

# Responses
text = response.output_text
```

### 9.3 认为每次响应都有文本

工具调用、拒答、截断和非文本任务可能没有普通文本。检查 `finish_reason`、`status`、`refusal`、`tool_calls`、`output` 和 `incomplete_details`。

### 9.4 认为 SDK 自动保存多轮对话

Chat Completions 的消息历史由应用传入；Responses 的 `previous_response_id` 使用服务端响应关联能力。二者都不等价于自动的用户会话数据库。

### 9.5 认为流式响应每一片都是文本

Chat 流的 `delta.content` 可能为空；Responses 流还包含开始、结束、失败等事件。按协议处理事件，不要对每个分片直接访问文本属性。

### 9.6 认为 SDK 有的方法服务端必然可用

`retrieve/list/update/delete` 等客户端方法遇到不兼容服务器可能返回 404/405。需要时先确认服务端支持、数据存储打开且当前账户有权限。

### 9.7 认为低 temperature 或 JSON Schema 能保证业务正确

模型仍可能遗漏事实、拒绝回答、输出业务上不允许的值，或在边界输入上失败。输出应通过类型和业务规则校验，必要时重试或交给人工处理。

## 10. 学习与练习顺序

1. 用 `chat.completions.create()` 完成一次非流式对话，并读取 `id/model/choices/usage`。
2. 实现三轮历史对话，助手答复追加回 `messages`。
3. 用 `stream=True` 显示增量文本并拼接最终答案。
4. 用 Pydantic 模型和 `chat.completions.parse()` 解析一份文本分类结果。
5. 在服务商支持的前提下，用 `responses.create()` 完成相同任务，比较请求字段和响应结构。
6. 用 `responses.stream()` 处理文本 delta 和终态事件。
7. 测试 `previous_response_id`、retrieve 或其他管理型方法前，先确认服务端是否支持存储。
8. 给调用加上超时、有限重试、异常分类和不包含敏感内容的日志。

## 11. 验收清单

- [ ] 能解释 `OpenAI -> 资源 -> 方法 -> 响应对象` 的调用层级。
- [ ] 能用环境变量创建客户端，并知道 `QWEN_API_KEY` 需要显式传入。
- [ ] 能使用 `chat.completions.create()` 并正确读取 `choices[0].message.content`。
- [ ] 能解释 `ChatCompletion` 的 `id/model/choices/usage` 及 Choice 的 `finish_reason`。
- [ ] 能用 `chat.completions.parse()` 得到并检查 `message.parsed`。
- [ ] 能处理 Chat 流的 `delta.content` 为空和流中断。
- [ ] 能使用 `responses.create()` 并正确读取 `output_text/status/output/usage`。
- [ ] 能解释 Responses output item 是类型化集合，不能固定取第一个文本字段。
- [ ] 能用 `responses.parse()` 获取 `output_parsed`。
- [ ] 能区分 Responses 事件流和 Chat Completion chunk 流。
- [ ] 理解 `previous_response_id`、`store` 和响应生命周期依赖服务端存储能力。
- [ ] 知道 `retrieve/list/update/delete/cancel/compact` 不能默认在第三方服务可用。
- [ ] 能处理认证、限流、超时、连接和参数错误，并且不记录真实密钥。

掌握这些内容后，可继续阶段 2 的结构化输出、工具调用与会话状态。方法签名和模型能力可能升级；遇到 SDK 升级时，以当前锁定版本、SDK 类型提示和服务商文档共同确认。
