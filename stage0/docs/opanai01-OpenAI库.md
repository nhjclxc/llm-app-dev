# `openai` Python 库：`OpenAI` 客户端必学方法和属性

> 适合对象：已经会 Python，正在学习大模型应用开发的初学者。
>
> 入口代码：
>
> ```python
> from openai import OpenAI
>
> client = OpenAI()
> ```
>
> 版本说明：当前仓库 `stage0` 虚拟环境中安装的是 `openai 3.16.2`。SDK 会持续变化，学习时应以自己环境中的版本、服务商接口文档和模型能力为准。本文重点讲稳定的调用结构，不要求一次记住所有资源。

## 1. 先记住一个核心结论

`OpenAI` 对象不是“一个直接生成文本的方法”，而是一个同步 API 客户端和资源入口。真正的业务调用通常位于多级属性之后：

```python
client.responses.create(...)
client.chat.completions.create(...)
client.embeddings.create(...)
client.models.list(...)
client.files.create(...)
```

可以把调用层级理解成：

```text
OpenAI 客户端
  ├── 配置和连接管理：api_key、base_url、timeout、max_retries
  ├── responses：新式文本、多模态、结构化输出、工具调用
  ├── chat.completions：兼容性较好的聊天接口
  ├── embeddings：文本向量
  ├── models：查看模型
  ├── files：上传和管理文件
  ├── vector_stores：托管向量库相关能力
  ├── batches：批量异步任务
  └── 其他资源：图片、音频、审核、实时通信等
```

初学者不要把所有 `client.xxx` 都当成必须背下来的 API。先理解请求链路：

```text
创建客户端
  -> 选择资源
  -> 调用资源方法
  -> 得到 SDK 类型化响应
  -> 读取响应属性
  -> 处理异常、超时、重试和资源关闭
```

## 2. 学习优先级

### 2.1 第一优先级：必须掌握

| 类型 | 必学内容 | 用途 |
| --- | --- | --- |
| 创建客户端 | `OpenAI(...)` | 配置 Key、地址、超时和重试 |
| 文本生成 | `client.responses.create(...)` | 新项目的主要文本调用入口 |
| 兼容聊天 | `client.chat.completions.create(...)` | 兼容 OpenAI 格式的第三方模型服务 |
| 响应读取 | `output_text`、`output`、`choices`、`usage`、`id`、`status` | 读取答案、状态和成本信息 |
| 错误处理 | `AuthenticationError`、`RateLimitError`、`APITimeoutError`、`APIConnectionError`、`APIStatusError` | 让程序可诊断、可恢复 |
| 请求控制 | `timeout`、`max_retries`、每次请求的 `timeout=` | 防止请求永久等待或无限重试 |
| 结构化输出 | `responses.parse(...)` 或 `chat.completions.parse(...)` | 将模型输出变成 Pydantic 对象 |
| 生命周期 | `close()`、`is_closed`、`with client:` | 正确释放 HTTP 连接 |
| 配置副本 | `with_options(...)` | 为单次任务创建不同配置的客户端 |

### 2.2 第二优先级：做 RAG 和应用时必须掌握

| 类型 | 必学内容 | 用途 |
| --- | --- | --- |
| 流式输出 | `responses.stream(...)`、`chat.completions.create(stream=True)` | 实时显示回答 |
| 向量 | `client.embeddings.create(...)` | RAG 的文本向量化 |
| 模型管理 | `client.models.list()`、`retrieve()` | 检查模型名称和可用性 |
| 工具调用 | `tools`、`tool_choice`、工具结果回传 | 让模型调用你的 Python 函数 |
| 文件 | `client.files.create/list/retrieve/content/delete` | 文件输入、批处理或托管检索 |
| 分页 | `for item in client.models.list()`、`has_next_page()` | 遍历列表资源 |

### 2.3 第三优先级：有实际需求再学习

- `with_raw_response`：需要状态码、响应头、原始 HTTP 响应时使用。
- `with_streaming_response`：需要自己管理底层流式 HTTP 响应时使用。
- `AsyncOpenAI`：FastAPI 或高并发异步服务需要时使用。
- `client.request/get/post/patch/put/delete`：官方资源不支持某个接口，或需要调用兼容服务商的自定义路径时使用。
- `vector_stores`、`batches`、`fine_tuning`、`webhooks`、`realtime`、`containers`、`evals`：对应业务出现后再学。

不要先学习以下内容：以下划线开头的私有方法、内部 HTTPX 包装类、生成代码的内部类型和所有资源的完整签名。它们不是大模型应用入门的主线。

## 3. 创建 `OpenAI` 客户端

### 3.1 最简单的写法

```python
from openai import OpenAI

client = OpenAI()
```

在当前 SDK 中，`OpenAI()` 会优先从环境变量读取默认配置。连接 OpenAI 官方服务时，至少需要：

```dotenv
OPENAI_API_KEY=你的密钥
```

如果没有提供 Key，也没有找到对应环境变量，客户端构造阶段就可能抛出缺少凭证的错误。

### 3.2 显式传入配置

更适合学习和排错的写法：

```python
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    timeout=30.0,
    max_retries=2,
)
```

显式传入的价值：

- 能清楚看到客户端使用哪个 Key。
- 可以为不同服务配置不同的 `base_url`。
- 可以控制请求超时和 SDK 自动重试次数。
- 出错时更容易判断是环境变量问题还是代码问题。

### 3.3 本仓库的通义兼容接口写法

当前仓库的 `core/model_helper.py` 使用的是通义兼容接口，并不是 OpenAI 默认地址。因此不能只写 `OpenAI()`，还需要显式传入 Key 和 `base_url`：

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
```

例如 `.env` 可以是：

```dotenv
QWEN_API_KEY=你的密钥
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=你的模型名称
```

注意：`OpenAI()` 自动查找的是 `OPENAI_API_KEY`，不会因为变量叫 `QWEN_API_KEY` 就自动读取它。第三方兼容服务还可能只实现 Chat Completions，不一定实现 Responses API；因此调用前要核对服务商的兼容范围。

### 3.4 `OpenAI` 构造函数参数

当前版本常用参数如下：

| 参数 | 是否初学必学 | 作用 |
| --- | --- | --- |
| `api_key` | 是 | API 认证密钥，也可以传入一个延迟获取 Key 的函数 |
| `base_url` | 是 | API 根地址；第三方 OpenAI 兼容服务必须重点检查 |
| `timeout` | 是 | 客户端默认超时时间；可用秒数或 HTTPX Timeout 对象 |
| `max_retries` | 是 | SDK 遇到部分临时错误时的自动重试次数，当前本地版本默认值为 2 |
| `default_headers` | 后续 | 给每个请求添加默认 HTTP Header |
| `default_query` | 后续 | 给每个请求添加默认查询参数 |
| `http_client` | 后续 | 注入自己配置的 HTTPX 客户端，例如代理、连接池或自定义传输 |
| `organization` | 后续 | OpenAI 组织标识 |
| `project` | 后续 | OpenAI 项目标识 |
| `webhook_secret` | 后续 | 验证 Webhook 签名 |
| `websocket_base_url` | 后续 | WebSocket/实时通信地址 |
| `admin_api_key` | 管理功能 | 管理 API 使用的密钥，不是普通聊天 Key |
| `provider`、`workload_identity`、`data_residency` | 高级 | 企业认证、身份工作负载和区域数据驻留场景 |

第一阶段只需要稳定掌握前四项。不要为了“把构造函数参数全部学完”而提前学习企业认证机制。

## 4. `client` 本身的必学属性和方法

### 4.1 `client.api_key`

```python
print(bool(client.api_key))
```

用途：确认客户端是否持有 Key。

安全要求：

- 不要 `print(client.api_key)`。
- 不要把它放到异常信息、日志、截图和 HTTP 响应中。
- 排错时只打印 `bool(client.api_key)` 或脱敏后的前几位。

当前仓库的 `stage0/main.py` 中如果保留了直接打印 `model_helper.qwen_api_key` 的代码，应删除或改成只打印是否存在。Key 只要进入终端日志，就可能被 shell 历史、录屏或日志采集保存。

### 4.2 `client.base_url`

```python
print(client.base_url)
```

用途：确认请求将发送到哪个 API 根地址。

常见问题：

- 把完整路径 `.../chat/completions` 错当成 `base_url`。
- 末尾版本路径写错，例如服务商要求 `/v1`，却少写或重复写 `/v1`。
- Key 属于 A 服务商，却把请求发到 B 服务商。

通常在创建客户端时配置它，不建议在运行过程中频繁修改：

```python
client = OpenAI(
    api_key=api_key,
    base_url="https://example.com/v1",
)
```

### 4.3 `client.timeout`

```python
print(client.timeout)
```

它是客户端级默认超时。单个请求可以覆盖它：

```python
response = client.responses.create(
    model=model,
    input="快速回答：什么是 Token？",
    timeout=10.0,
)
```

建议：

- 命令行脚本设置有限超时，避免网络异常时一直卡住。
- 文档解析、复杂推理等任务可以单独设置更长超时。
- 超时不是模型“回答太慢”的唯一原因，也可能是 DNS、代理、连接池或服务端排队。

### 4.4 `client.max_retries`

```python
print(client.max_retries)
```

它控制 SDK 层面的自动重试次数。重试不能代替业务层的可靠性设计：

- 记录每次请求的开始时间、结束时间和最终异常。
- 对用户可见的请求设置总耗时上限。
- 对有副作用的工具调用，不能因为模型或网络重试就重复执行写操作。
- 对限流要结合退避、队列和并发控制。

创建客户端时设置：

```python
client = OpenAI(
    api_key=api_key,
    timeout=30.0,
    max_retries=2,
)
```

### 4.5 `client.default_headers` 和 `client.default_query`

用途：为每次请求添加默认 Header 或查询参数。

```python
client = OpenAI(
    api_key=api_key,
    default_headers={
        "X-Client-Name": "llm-app-dev",
    },
    default_query={
        "api-version": "2026-01-01",
    },
)
```

更常见的是只对一次请求添加参数：

```python
response = client.responses.create(
    model=model,
    input="你好",
    extra_headers={"X-Trace-ID": trace_id},
    extra_query={"debug": "false"},
    timeout=15.0,
)
```

不要随意覆盖 `Authorization`、`Content-Type` 等 SDK 管理的 Header，除非你明确知道兼容服务商的认证协议。

### 4.6 `client.organization`、`client.project`、`client.webhook_secret`

这些属性主要用于 OpenAI 平台的组织、项目和 Webhook 场景。普通聊天应用通常不需要主动设置，但要知道它们可能由环境变量自动读取：

```dotenv
OPENAI_ORG_ID=...
OPENAI_PROJECT_ID=...
OPENAI_WEBHOOK_SECRET=...
```

不要把它们和普通 API Key 混为一谈：

- `api_key` 用于认证请求。
- `organization` 和 `project` 用于指定请求归属。
- `webhook_secret` 用于验证服务端收到的 Webhook 请求是否来自可信发送方。

### 4.7 `client.auth_headers`、`client.user_agent`

这类属性主要用于诊断：

```python
print(client.user_agent)
print(client.auth_headers.keys())
```

不要打印 `client.auth_headers` 的完整值，因为里面包含 `Authorization`。如果需要排查认证，只打印 Header 名称，或确认是否存在 `Authorization` 键。

### 4.8 `client.is_closed()`、`client.close()` 和上下文管理器

```python
client = OpenAI(api_key=api_key)
try:
    response = client.responses.create(
        model=model,
        input="你好",
    )
finally:
    client.close()

print(client.is_closed())
```

更推荐在短生命周期脚本中使用：

```python
from openai import OpenAI


with OpenAI(api_key=api_key) as client:
    response = client.responses.create(
        model=model,
        input="你好",
    )
    print(response.output_text)
```

在长生命周期的 Web 服务中，通常在应用启动时创建一个共享客户端，在应用关闭时关闭它，不要每个请求都创建一个新客户端。

## 5. 核心资源属性

### 5.1 `client.responses`

这是新项目优先学习的文本/多模态响应入口。最重要的方法：

| 方法 | 作用 |
| --- | --- |
| `create()` | 创建一次模型响应 |
| `parse()` | 创建响应并把结构化输出解析成指定 Python 类型 |
| `stream()` | 以事件流方式读取响应 |
| `retrieve()` | 根据响应 ID 查询已有响应 |
| `cancel()` | 取消后台响应 |
| `delete()` | 删除响应记录 |
| `compact()` | 对上下文进行压缩，属于进阶上下文管理 |

最小调用：

```python
response = client.responses.create(
    model=model,
    instructions="你是一个简洁的中文学习助手。",
    input="用两句话解释大模型应用开发。",
)

print(response.output_text)
print(response.id)
print(response.status)
print(response.usage)
```

最常用的 `create()` 参数：

| 参数 | 作用 |
| --- | --- |
| `model` | 指定模型 |
| `input` | 文本、图片、文件或输入项 |
| `instructions` | 系统/开发者级指令 |
| `max_output_tokens` | 限制输出 Token 上限 |
| `temperature` | 控制随机性，是否支持取决于模型 |
| `tools` | 声明模型可使用的工具 |
| `tool_choice` | 控制工具选择策略 |
| `previous_response_id` | 关联前一响应，实现连续上下文 |
| `conversation` | 使用服务端会话资源 |
| `store` | 控制响应是否保存，具体行为看接口和服务策略 |
| `metadata` | 附加业务元数据，便于检索或观测 |
| `timeout` | 只覆盖本次请求的超时时间 |

不要一次把所有参数都设置上。先固定 `model`、`input`、`instructions` 和 `max_output_tokens`，确认任务稳定后再增加工具、会话和观测参数。

### 5.2 `response` 的必学属性

```python
response = client.responses.create(
    model=model,
    input="介绍 Python 的列表。",
)

print(response.output_text)  # 聚合后的文本
print(response.output)       # 完整输出项，工具调用时必须查看
print(response.id)           # 响应 ID
print(response.model)        # 实际使用的模型
print(response.status)       # completed、failed、incomplete 等状态
print(response.usage)        # 输入、输出和总 Token
print(response.error)        # 生成失败时的错误信息
```

初学者最容易犯的错误是只取 `response.output[0]`。Responses 的 `output` 可能包含文本消息、工具调用、推理相关项目或其他输出项。只需要最终文本时用 `response.output_text`；需要实现工具调用或调试完整链路时再查看 `response.output`。

### 5.3 `client.chat.completions`

这是当前仓库已有代码使用的调用方式，也是许多第三方 OpenAI 兼容服务首先支持的接口。它的核心方法：

| 方法 | 作用 |
| --- | --- |
| `create()` | 创建聊天补全 |
| `parse()` | 将结构化输出解析成 Pydantic 对象 |
| `stream()` | 以流式方式获取聊天事件 |
| `list()` | 列出已保存的聊天补全，是否可用取决于服务 |
| `retrieve()` | 查询某个补全 |
| `update()` | 更新某个补全的元数据 |
| `delete()` | 删除某个补全 |

最小调用：

```python
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "你是简洁的中文助手。"},
        {"role": "user", "content": "用两句话解释 Token。"},
    ],
    temperature=0.2,
)

message = completion.choices[0].message
print(message.content)
print(completion.id)
print(completion.usage)
```

### 5.4 `ChatCompletion` 响应结构

```text
completion
  ├── id
  ├── model
  ├── choices
  │     └── choices[0]
  │           ├── index
  │           ├── message
  │           │     ├── role
  │           │     ├── content
  │           │     └── tool_calls
  │           └── finish_reason
  └── usage
```

常用读取方式：

```python
choice = completion.choices[0]

print(choice.message.role)
print(choice.message.content)
print(choice.finish_reason)
print(completion.usage.prompt_tokens)
print(completion.usage.completion_tokens)
print(completion.usage.total_tokens)
```

如果模型返回工具调用，`message.content` 可能为空，需要读取：

```python
for tool_call in choice.message.tool_calls or []:
    print(tool_call.id)
    print(tool_call.function.name)
    print(tool_call.function.arguments)
```

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

### 5.6 `client.embeddings`

核心方法只有：

```python
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["第一段文本", "第二段文本"],
)

for item in embedding_response.data:
    print(item.index)
    print(len(item.embedding))
```

必须理解的属性：

```python
print(embedding_response.model)
print(embedding_response.data[0].embedding)
print(embedding_response.usage)
```

重要知识点：

- `input` 可以是一段文本，也可以是多段文本的列表。
- Embedding 是向量，不是给用户看的答案。
- 同一个向量库必须保持 Embedding 模型和维度的一致性。
- 输入文本不能无限长；文档需要先切块和清洗。
- 向量化批量请求时，仍要关注单次请求 Token 总量和服务商限制。

在本仓库中，通义 Embedding 使用 `DashScopeEmbeddings`，并不一定通过 `client.embeddings.create()` 调用。学习时要区分：OpenAI SDK 的 Embedding 资源和 LangChain/服务商自己的 Embedding 封装。

### 5.7 `client.models`

三个重要方法：

```python
models = client.models.list()
for item in models:
    print(item.id, item.owned_by)

model_info = client.models.retrieve(model)
print(model_info.id)
```

用途：

- 检查 Key 是否能访问模型列表。
- 确认模型名称没有拼错。
- 在诊断 404 或“模型不存在”时提供证据。

不要仅因为模型出现在列表中，就假设它支持所有能力。文本、视觉、工具、Embedding、音频和结构化输出能力仍要看模型文档。

### 5.8 `client.files`

常用方法：

```python
uploaded = client.files.create(
    file=open("data/example.pdf", "rb"),
    purpose="user_data",
)

info = client.files.retrieve(uploaded.id)
content = client.files.content(uploaded.id)

for file_info in client.files.list():
    print(file_info.id, file_info.filename)

client.files.delete(uploaded.id)
```

注意：

- 文件上传是网络 I/O，必须处理超时和失败。
- 不要把用户上传的文件路径、内容和 API 响应直接写入公共日志。
- 删除本地文件不等于删除服务端文件。
- `purpose` 会影响文件的使用场景和服务端处理方式。
- 做 RAG 时要区分“上传文件”“建立索引”“检索文件内容”三个步骤。

### 5.9 `client.vector_stores`

这是服务端托管检索相关资源，常见方法：

```python
vector_store = client.vector_stores.create(
    name="my-knowledge-base",
)

print(vector_store.id)
print(client.vector_stores.retrieve(vector_store.id))
print(client.vector_stores.search(vector_store.id, query="如何退款？"))
```

初学 RAG 时不必立即使用它。先用本地文档、Embedding 和一个向量数据库理解“加载、切块、向量化、检索、引用”的完整链路，再学习托管 Vector Store 的生命周期和权限。

### 5.10 `client.images`、`client.audio`、`client.moderations`

它们属于按业务选择的资源：

```python
image = client.images.generate(
    model="image-model",
    prompt="一张简洁的技术架构图",
)

speech = client.audio.speech.create(
    model="audio-model",
    voice="alloy",
    input="你好，这是语音测试。",
)

moderation = client.moderations.create(
    model="moderation-model",
    input="待审核文本",
)
```

这些调用的模型、参数和返回结构差异较大，不属于文本聊天入门的必学内容。只有项目需要图像、语音或内容安全时再单独学习。

### 5.11 `client.batches`

Batch 用于批量、异步处理，常见方法：

```python
batch = client.batches.create(
    input_file_id="file-id",
    endpoint="/v1/responses",
    completion_window="24h",
)

current = client.batches.retrieve(batch.id)
print(current.status)
```

它适合离线评测、批量 Embedding 或批量数据处理，不适合直接替代用户实时聊天请求。必须理解任务状态、结果文件、失败任务和取消任务。

## 6. 响应对象的通用操作

### 6.1 属性访问和序列化

SDK 返回的对象通常可以使用属性访问：

```python
print(response.id)
print(response.output_text)
```

需要保存到 JSON 或日志时，可以使用 Pydantic 模型的序列化方法：

```python
payload = response.model_dump()
```

日志不要无条件保存完整请求和响应。应根据数据敏感等级决定是否脱敏、截断或只保留摘要。

### 6.2 `usage` 和成本观测

文本响应成功后，应至少记录：

```python
usage = response.usage
if usage:
    print(usage.input_tokens)
    print(usage.output_tokens)
    print(usage.total_tokens)
```

Chat Completions 的字段名称通常是 `prompt_tokens`、`completion_tokens` 和 `total_tokens`。不要把两种响应结构写成同一个硬编码路径，应在适配层统一成自己的业务结构。

推荐业务日志字段：

```text
request_id
model
started_at
finished_at
latency_ms
input_tokens
output_tokens
total_tokens
status
error_type
```

### 6.3 分页结果

很多 `list()` 方法返回的是分页对象，而不是普通列表：

```python
page = client.models.list()

for model in page:
    print(model.id)
```

需要手动控制分页时：

```python
page = client.models.list()
while True:
    for model in page:
        print(model.id)

    if not page.has_next_page():
        break
    page = page.get_next_page()
```

不要假设一次 `list()` 就能拿到所有数据；数据量大时要设置上限、分页和超时。

## 7. 结构化输出：`parse()`

当模型输出需要被程序继续使用时，不要依赖字符串切分。可以使用 Pydantic 定义结构。

### 7.1 Chat Completions 结构化输出

```python
from pydantic import BaseModel


class Ticket(BaseModel):
    category: str
    priority: str
    summary: str


parsed_completion = client.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": "分析工单并按给定结构返回。"},
        {"role": "user", "content": "用户反馈：支付后订单状态仍未更新。"},
    ],
    response_format=Ticket,
)

message = parsed_completion.choices[0].message
if message.parsed:
    ticket: Ticket = message.parsed
    print(ticket.category)
    print(ticket.priority)
    print(ticket.summary)
else:
    print(message.refusal)
```

### 7.2 Responses 结构化输出

```python
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    confidence: float


parsed_response = client.responses.parse(
    model=model,
    instructions="返回结构化答案，不要输出额外解释。",
    input="Python 的字典如何读取不存在的键？",
    text_format=Answer,
)

answer: Answer | None = parsed_response.output_parsed
if answer is not None:
    print(answer.answer)
    print(answer.confidence)
```

结构化输出仍然要做业务校验：

- `confidence` 是否在 0 到 1 之间。
- `category` 是否属于业务允许的枚举值。
- 关键字段为空时是否需要重试或转人工。
- 模型拒答时是否有 `refusal` 或错误状态。

## 8. 流式输出

流式输出的目标是降低用户等待完整结果的感受，不代表模型更快完成，也不代表可以跳过错误处理。

### 8.1 Responses 流式调用

```python
with client.responses.stream(
    model=model,
    instructions="你是简洁的中文助手。",
    input="用三句话解释什么是 RAG。",
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "response.completed":
            print("\n[完成]")

    final_response = stream.get_final_response()
    print(final_response.usage)
```

这里要理解两个对象：

- `stream`：事件迭代器，逐步接收事件。
- `final_response`：流读取完成后聚合出的最终响应。

### 8.2 Chat Completions 流式调用

```python
stream = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "解释一下上下文窗口。"},
    ],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

流式代码必须考虑：

- 中途连接断开后如何提示用户。
- 已输出部分内容后不能假装完整成功。
- 使用工具调用时，不能把工具参数分片直接执行。
- 是否能获得完整 Token 用量，取决于接口和请求配置。

## 9. 工具调用：方法是 `create()`，能力来自 `tools`

工具调用不是 `client` 上一个叫 `call_tool()` 的方法，而是把工具描述传给模型，模型返回工具调用参数，应用验证参数并执行 Python 函数。

流程：

```text
你的代码定义工具 Schema
  -> client.responses.create(tools=[...])
  -> 模型返回 function call
  -> 你的代码校验参数
  -> 你的代码执行函数
  -> 把工具结果回传给模型
  -> 模型生成最终回答
```

一个简化的 Chat Completions 工具声明：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "查询订单状态，只允许查询，不允许修改订单。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    }
]

completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "查询订单 A100 的状态。"}],
    tools=tools,
    tool_choice="auto",
)
```

必须遵守：

- 模型只负责提出调用意图，真正的函数由你的程序执行。
- 先用 Pydantic 或 JSON Schema 校验参数，再访问数据库或外部 API。
- 读操作和写操作分开授权。
- 删除、付款、发送消息等高风险操作必须人工确认。
- 设置最大工具调用次数、超时和总 Token 预算。
- 不允许模型直接生成 Shell 命令、SQL 或任意 URL 后执行。

## 10. 错误处理

### 10.1 常用异常层级

```text
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
```

### 10.2 推荐的最小处理方式

```python
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)


try:
    response = client.responses.create(
        model=model,
        input="你好",
        timeout=20.0,
    )
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
```

捕获顺序很重要：`APITimeoutError` 是连接错误的一种，应放在 `APIConnectionError` 前面；具体状态码异常应放在通用 `APIStatusError` 前面。

### 10.3 错误对象中值得记录的属性

| 属性 | 用途 | 是否可以直接写日志 |
| --- | --- | --- |
| `status_code` | 判断 4xx/5xx | 可以 |
| `request_id` | 向服务商排查单次请求 | 可以，注意脱敏策略 |
| `body` | 查看服务端错误细节 | 需要过滤敏感内容 |
| `code` | 业务错误码 | 可以 |
| `param` | 指出错误参数 | 可以 |
| `type` | 服务端错误类型 | 可以 |
| `request` | 底层 HTTP 请求 | 不要无条件完整打印，可能包含敏感 Header |

## 11. `with_options()`：为单个场景派生客户端

不要为了一个请求修改全局客户端配置。可以创建一个继承原配置的新客户端：

```python
default_client = OpenAI(
    api_key=api_key,
    timeout=30.0,
    max_retries=2,
)

slow_client = default_client.with_options(
    timeout=120.0,
    max_retries=1,
)

response = slow_client.responses.create(
    model=model,
    input="处理一份较长文档。",
)
```

常见用途：

- 普通聊天使用 30 秒，文档处理使用 120 秒。
- 评测任务关闭或减少自动重试。
- 为某个租户增加 Header 或查询参数。
- 在同一个进程中使用不同的 `base_url`。

`with_options()` 返回新的客户端配置，不要把它误解为只修改当前客户端的临时上下文。长时间运行的程序仍需合理关闭派生客户端和底层 HTTP 资源；简单项目优先复用一个共享客户端。

## 12. `with_raw_response` 和 `with_streaming_response`

### 12.1 读取原始响应

```python
raw_response = client.with_raw_response.responses.create(
    model=model,
    input="你好",
)

print(raw_response.status_code)
print(raw_response.headers)
parsed_response = raw_response.parse()
print(parsed_response.output_text)
```

什么时候使用：

- 需要读取响应头中的请求 ID、限流信息或缓存信息。
- 需要记录 HTTP 状态码。
- 需要调试第三方兼容服务返回的额外 Header。

普通业务代码优先使用已经解析好的 `response`，不要为了“看起来更底层”而所有请求都走 raw response。

### 12.2 流式 HTTP 响应

`with_streaming_response` 适合需要自己控制读取和关闭底层 HTTP 流的高级场景。初学者先掌握 `client.responses.stream(...)`，后续再学习它。

## 13. 异步客户端 `AsyncOpenAI`

当你开始使用 FastAPI、异步任务或需要并发处理多个请求时，再学习异步版本：

```python
from openai import AsyncOpenAI


async_client = AsyncOpenAI(api_key=api_key)

response = await async_client.responses.create(
    model=model,
    input="你好",
)

print(response.output_text)
await async_client.close()
```

异步规则：

- `AsyncOpenAI` 的网络方法需要 `await`。
- 异步流使用 `async for`。
- 关闭客户端使用 `await async_client.close()`。
- 不要在同步函数里混用异步客户端，也不要在每次请求中创建新的异步客户端。

## 14. 底层 HTTP 方法：知道存在，但不要先学

`OpenAI` 继承了底层客户端能力，常见方法包括：

```python
client.request(...)
client.get(...)
client.post(...)
client.patch(...)
client.put(...)
client.delete(...)
```

它们适用于：

- SDK 尚未生成某个新接口。
- 第三方兼容服务提供了官方 SDK 没有封装的路径。
- 需要完全控制返回类型、原始请求和自定义 Body。

但这些方法通常需要 SDK 内部的 `RequestOptions`、响应类型和转换逻辑。初学者直接使用它们，容易绕过认证、错误转换、类型解析和重试行为。

建议顺序：

1. 先查是否存在 `client.responses`、`client.files` 等资源。
2. 再查资源下是否有 `create/list/retrieve/update/delete` 方法。
3. 确实没有封装时，再研究底层 HTTP 方法。
4. 将自定义调用放在自己的适配层，不要在业务代码中到处拼接 URL。

## 15. 常见错误和正确理解

### 错误 1：以为 `OpenAI()` 自动读取所有变量

错误理解：设置了 `QWEN_API_KEY`，就可以直接 `OpenAI()`。

正确理解：默认环境变量名称和服务商有关。当前 OpenAI SDK 默认读取 `OPENAI_API_KEY`；本仓库的通义变量是 `QWEN_API_KEY`，需要显式传 `api_key`，通常还需要显式传 `base_url`。

### 错误 2：把 `client.chat` 当成可以直接调用的函数

错误：

```python
client.chat("你好")
```

正确：

```python
client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "你好"}],
)
```

### 错误 3：混用两种响应读取方式

错误：

```python
response = client.responses.create(...)
print(response.choices[0].message.content)
```

正确：

```python
response = client.responses.create(...)
print(response.output_text)
```

Chat Completions 才使用：

```python
completion.choices[0].message.content
```

### 错误 4：每次请求都创建客户端

错误：

```python
def ask(question: str) -> str:
    client = OpenAI(api_key=api_key)
    return client.responses.create(model=model, input=question).output_text
```

长期运行的服务应在应用启动时创建共享客户端，复用连接池；单次脚本可以使用上下文管理器。

### 错误 5：捕获 `Exception` 后丢失信息

错误：

```python
try:
    ...
except Exception:
    return "调用失败"
```

这样用户和开发者都不知道失败原因。应至少区分认证、网络、超时、限流、参数错误和服务端错误，并记录请求 ID。

### 错误 6：把自动重试当成幂等保证

SDK 重试网络请求并不意味着你的业务写操作一定不会重复。涉及订单、扣费、删除、发送消息的工具必须有幂等键、状态检查和人工确认。

## 16. 必学代码模板

下面模板适合先保存为 `stage0/openai_client_demo.py`，然后根据当前服务商修改环境变量和模型名：

```python
import os

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)


load_dotenv()


def create_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=30.0,
        max_retries=2,
    )


def ask(client: OpenAI, question: str) -> str:
    response = client.responses.create(
        model=os.environ["OPENAI_MODEL"],
        instructions="你是简洁、准确的中文学习助手。",
        input=question,
        max_output_tokens=300,
    )
    return response.output_text


def main() -> None:
    client = create_client()
    try:
        answer = ask(client, "什么是大模型应用开发？")
        print(answer)
    except AuthenticationError:
        print("认证失败，请检查 OPENAI_API_KEY。")
    except RateLimitError:
        print("请求被限流，请降低调用频率。")
    except APITimeoutError:
        print("请求超时，请检查网络或提高 timeout。")
    except APIConnectionError:
        print("连接失败，请检查网络、代理和 base_url。")
    except APIStatusError as error:
        print(f"服务端错误：status={error.status_code}, request_id={error.request_id}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
```

如果当前模型服务只支持 Chat Completions，把 `ask()` 中的调用改为：

```python
completion = client.chat.completions.create(
    model=os.environ["OPENAI_MODEL"],
    messages=[
        {"role": "system", "content": "你是简洁、准确的中文学习助手。"},
        {"role": "user", "content": question},
    ],
    max_completion_tokens=300,
)
return completion.choices[0].message.content or ""
```

## 17. 推荐学习顺序和练习

### 练习 1：打印客户端配置，但不泄露密钥

```python
print(client.base_url)
print(client.timeout)
print(client.max_retries)
print(bool(client.api_key))
```

### 练习 2：完成一次 Responses 调用

要求记录：`response.id`、`response.status`、`response.output_text` 和 `response.usage`。

### 练习 3：完成一次 Chat Completions 调用

要求正确读取：`completion.choices[0].message.content`，并和 Responses 的响应结构做对比。

### 练习 4：用 `models.list()` 排查模型名称

不要直接猜模型名。先列出可访问模型，再选择一个模型做请求。

### 练习 5：制造四种错误

- 删除 API Key。
- 使用错误的 `base_url`。
- 使用不存在的模型名。
- 将超时设置得非常短。

为每种错误记录异常类型、状态码和用户可见提示。

### 练习 6：完成一次结构化输出

用 Pydantic 定义工单结构，分别尝试 `chat.completions.parse()` 和 `responses.parse()`，理解 `message.parsed` 与 `response.output_parsed` 的区别。

### 练习 7：完成一次流式输出

先实现 Chat Completions 的 `stream=True`，再实现 Responses 的 `responses.stream()`，记录两种事件结构差异。

## 18. 本阶段验收清单

- [ ] 能解释 `OpenAI` 客户端、资源属性和资源方法的关系。
- [ ] 能正确配置 `api_key`、`base_url`、`timeout` 和 `max_retries`。
- [ ] 知道 `OpenAI()` 默认读取哪个环境变量，以及第三方兼容服务为什么可能需要显式配置。
- [ ] 能完成 `client.responses.create()` 并读取 `output_text`、`id`、`status`、`usage`。
- [ ] 能完成 `client.chat.completions.create()` 并读取 `choices[0].message.content`。
- [ ] 能区分 Responses 和 Chat Completions 的响应结构。
- [ ] 能解释 `client.embeddings.create()` 的用途，并知道它不是聊天接口。
- [ ] 能使用 `models.list()` 和分页结果。
- [ ] 能为客户端设置超时、重试并按异常类型处理错误。
- [ ] 能使用 `with client:` 或 `client.close()` 释放连接。
- [ ] 知道 `with_options()`、`with_raw_response`、`AsyncOpenAI` 分别解决什么问题。
- [ ] 没有打印、提交或记录 API Key。

## 19. 参考当前仓库的文件

- [阶段 0 学习文档](stage0/stage0-llm-app-roadmap.md)：环境、API 安全和最小调用。
- [`stage0/main.py`](stage0/main.py)：当前 Chat Completions 示例；其中的 Key 打印代码不应保留。
- [`core/model_helper.py`](core/model_helper.py)：仓库当前的 OpenAI 兼容客户端和 DashScope Embedding 封装。
- [`stage0/pyproject.toml`](stage0/pyproject.toml)：当前 `stage0` 项目的 OpenAI SDK 依赖声明。

完成本文件的“本阶段验收清单”后，再继续学习总路线中的 Prompt、流式输出、结构化输出和工具调用。
