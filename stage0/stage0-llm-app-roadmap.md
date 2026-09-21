# 阶段 0：环境、API 安全和开发习惯

> 对应总路线：[大模型应用开发学习路线](../llm-app-roadmap.md)
>
> 建议时长：3 到 5 天，每天 1.5 到 3 小时。
>
> 完成标准：你能在本地安全地运行一个 Python 大模型调用程序，理解一次请求的基本组成，并能独立排查常见问题。

## 1. 本阶段不学习什么

本阶段不学习模型训练、微调、RAG、向量数据库、Agent、LangChain 工作流和 Web 前端。这些内容依赖稳定的模型调用基础，过早引入只会让排错范围扩大。

本阶段只解决四个问题：

1. Python 项目和依赖如何稳定运行。
2. 模型 API 的请求和回复是什么。
3. API Key 怎样安全配置。
4. 遇到错误时怎样判断是代码、配置、网络还是账户问题。

## 2. 开始前的检查

在仓库根目录执行：

```bash
pwd
python3 --version
uv --version
git status --short
```

预期：

- 当前目录是 `llm-app-dev`。
- Python 建议为 3.11 或以上。
- `uv` 可用；若没有安装，先按 `环境搭建.md` 的说明安装。
- `git status --short` 中不应出现真实 API Key 或个人文档。

### 2.1 认识当前仓库

当前可参考的文件：

| 文件 | 用途 | 此阶段如何使用 |
| --- | --- | --- |
| `环境搭建.md` | 常用依赖和运行命令 | 作为安装命令参考，不必一次安装全部依赖 |
| `core/model_helper.py` | 聊天模型和 Embedding 客户端辅助函数 | 阅读其参数；先自行完成最小 API 调用后再复用 |
| `.gitignore` | Git 忽略规则 | 确认 `.env`、虚拟环境和本地数据不会被提交 |

不要修改已有的 `core/model_helper.py` 来完成本阶段练习。先在 `stage0/` 下写独立、短小、可删除的示例，出现问题时更容易定位。

## 3. Day 1：创建可重复的 Python 环境

### 3.1 为什么使用虚拟环境

不同项目依赖的 Python 包版本可能不同。虚拟环境把当前仓库的包隔离到 `.venv/`，避免升级一个包影响其他项目。

`uv` 负责创建环境、安装包和执行 Python。下面的命令应该始终在仓库根目录执行：

```bash
uv init
uv venv --python 3.11
uv add openai python-dotenv
uv add openai python-dotenv
```

含义：

- `uv venv --python 3.11`：创建 `.venv`，指定解释器版本。
- `uv add openai python-dotenv`：安装官方 OpenAI Python SDK 与读取 `.env` 文件的工具，并记录项目依赖。

之后运行代码优先使用：

```bash
uv run python main.py
```

### 3.3 创建学习目录

建议本阶段最终目录如下：

```text
stage0/
  stage0-llm-app-roadmap.md
  hello_llm.py
  .env.example
  notes.md
```

- `hello_llm.py`：最小模型调用程序。
- `.env.example`：变量名示例，可以提交到 Git。
- `notes.md`：记录命令、报错、耗时和自己的理解。

真实 `.env` 放在仓库根目录，或放在本地且已被 `.gitignore` 忽略的位置。它绝不能提交。

### Day 1 完成检查

- [ ] `uv --version` 可执行。
- [ ] `.venv/` 已创建，且不出现在 `git status --short` 中。
- [ ] `uv run python --version` 显示预期 Python 版本。
- [ ] `openai` 和 `python-dotenv` 已安装。

## 4. Day 2：理解大模型 API 的最小概念

### 4.1 API 调用的角色分工

一次聊天类模型调用通常包含以下部分：

| 概念 | 作用 | 例子 |
| --- | --- | --- |
| API Key | 证明调用者身份并计费 | `QWEN_API_KEY` |
| Base URL | API 服务地址 | OpenAI 兼容服务的 `/v1` 地址 |
| Model | 本次请求使用的模型名称 | `qwen3.6-plus` |
| System message | 定义模型的全局职责和规则 | “你是简洁的中文助手” |
| User message | 用户本次提出的问题 | “解释什么是 Token” |
| Response | 模型返回的内容和用量信息 | 文本、工具调用、Token 用量 |

模型不是数据库，也不会自动获得你电脑里的文件、互联网实时内容或你的业务系统数据。它只能处理你在请求中提供给它的内容，或通过程序明确允许的工具获取的信息。

### 4.2 Token 是什么

Token 是模型读取和生成文本时使用的计量单位，不等同于字符、汉字或单词。它影响：

- 成本：输入和输出通常分别计费。
- 上下文窗口：一次请求中可放入的总内容有限。
- 延迟：输入越长、输出越长，通常越慢。
- 截断：超出模型限制时请求可能失败，或输入/输出被截断。

现在不需要背不同模型的具体上限。需要养成习惯：任何多轮对话、文档问答或 Agent 都要考虑“本次到底向模型发送了多少内容”。

### 4.3 生成参数的基本认识

先只认识，不要在一个项目中同时调很多参数：

| 参数 | 影响 | 初学建议 |
| --- | --- | --- |
| `model` | 能力、速度、价格和上下文窗口 | 固定一个模型先跑通 |
| `temperature` | 输出随机性 | 提取、分类使用低值；创作可适当提高 |
| `max_tokens` 或同类参数 | 最大输出长度 | 为摘要、分类等任务设置合理上限 |
| `stream` | 是否分片返回 | 第一天先关闭，后续阶段再学习 |

即使将 `temperature` 设置得很低，模型也不能保证事实正确、格式永远合法或每次逐字相同。因此业务程序仍然需要校验和异常处理。

### Day 2 完成检查

- [ ] 能用自己的话解释 API Key、Base URL、Model、System message 和 User message。
- [ ] 能解释为什么长文档不能无限塞进一次请求。
- [ ] 知道模型输出是概率生成，不是可信数据库查询结果。

## 5. Day 3：安全配置 API Key

### 5.1 API Key 的规则

把 API Key 当作密码处理：拥有它的人可能以你的账户发起调用并产生费用。

必须遵守：

- 不写入 `.py`、`.md`、截图、聊天记录或 Git commit。
- 不在报错日志中打印完整 Key。
- 不在浏览器前端代码中配置服务端 API Key。
- 怀疑泄漏时立即去模型服务商控制台撤销或轮换，不是只删除本地文件。

### 5.2 创建真实环境文件和示例文件

在仓库根目录创建本地 `.env`：

```dotenv
QWEN_API_KEY=替换为真实密钥
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.6-plus
```

创建可提交的 `stage0/.env.example`：

```dotenv
QWEN_API_KEY=
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.6-plus
```

`.env.example` 中只出现变量名和非敏感默认值，绝不填真实 Key。

### 5.3 验证 Git 忽略规则

执行：

```bash
git check-ignore -v .env
git status --short
```

预期：第一条命令显示一条 `.gitignore` 规则，第二条命令不显示 `.env`。

若 `.env` 已经被 Git 跟踪，单纯加忽略规则没有用。不要直接假设它安全，先检查：

```bash
git ls-files .env
```

若输出 `.env`，说明它曾被跟踪。确认密钥已轮换后，再由你或仓库维护者按团队流程停止跟踪并清理历史泄漏风险。

### 5.4 `.env` 如何被 Python 读取

最简单的写法：

```python
import os

from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("QWEN_API_KEY")
```

`load_dotenv()` 会从当前工作目录及其父目录查找 `.env`。为了避免因从不同目录启动程序造成歧义，后续项目应明确约定：命令都在仓库根目录执行，真实 `.env` 放在仓库根目录。

不要这样写：

```python
api_key = "sk-真实密钥"
```

### Day 3 完成检查

- [ ] `.env` 已配置真实 Key，且不会出现在 Git 状态中。
- [ ] `stage0/.env.example` 不包含任何真实密钥。
- [ ] 程序能从 `os.environ` 读取 Key。
- [ ] 知道发现泄漏后应先轮换密钥。

## 6. Day 4：编写最小模型调用程序

先创建 `stage0/hello_llm.py`。下面示例采用 OpenAI 兼容客户端和当前仓库环境变量约定：

```python
import os
import sys

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI


load_dotenv()

api_key = os.environ.get("QWEN_API_KEY")
base_url = os.environ.get("QWEN_BASE_URL")
model = os.environ.get("QWEN_MODEL", "qwen3.6-plus")

if not api_key:
    sys.exit("未找到 QWEN_API_KEY：请在仓库根目录的 .env 中配置它。")

client = OpenAI(api_key=api_key, base_url=base_url)

try:
    response = client.chat.completions.create(
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

print(response.choices[0].message.content)
```

运行：

```bash
uv run python stage0/hello_llm.py
```

### 6.1 逐行理解关键代码

- `load_dotenv()`：读取本地环境文件，使变量可通过 `os.environ` 获取。
- `os.environ.get(...)`：读取配置；缺失时返回 `None` 或提供的默认值。
- `OpenAI(...)`：创建 API 客户端。OpenAI 兼容服务通常通过 `base_url` 指向服务商地址。
- `messages`：本次请求的对话内容；系统消息在前，用户任务在后。
- `client.chat.completions.create(...)`：发送请求并等待完整结果。
- `response.choices[0].message.content`：读取第一条候选回复的文本内容。
- `except`：将认证、连接和服务端错误转换为可理解的信息。

### 6.2 第一次运行时观察什么

不要只看“有一段回答”就结束。观察并记录：

- 从开始运行到看到回复大约多久。
- 输出是否符合“两句话”和“中文”的约束。
- `model` 改为不存在名称时的报错。
- `QWEN_API_KEY` 清空时的报错。
- 网络断开或代理错误时的报错。

在 `stage0/notes.md` 中记录日期、命令、模型名、输入、结果和错误原因。这个习惯会在后续 RAG 和 Agent 排错时发挥作用。

### Day 4 完成检查

- [ ] `hello_llm.py` 可打印模型回复。
- [ ] 未设置 Key 时，程序不发请求且显示明确提示。
- [ ] Key 错误、网络错误、服务错误的提示不同。
- [ ] 没有在终端或笔记中记录完整 Key。

## 7. Day 5：建立基础开发习惯

### 7.1 每次运行前后做的事

运行前：

1. 确认在仓库根目录。
2. 确认使用 `uv run main.py`，避免误用系统 Python。
3. 确认 `.env` 不会被 Git 跟踪。
4. 明确本次测试的输入和预期输出。

运行后：

1. 记录模型名、关键参数、输入大小和耗时。
2. 保留错误信息，但遮盖 API Key、用户隐私和敏感文本。
3. 用 `git diff` 检查改动只包含预期文件。
4. 用 `git status --short` 再确认没有敏感文件。

### 7.2 为模型调用设置边界

从第一个程序开始保留这些原则：

- 用户输入是外部不可信数据，不能直接让它修改系统规则。
- 模型输出也是外部不可信数据，不能直接执行为 Shell、SQL 或文件操作。
- 模型不知道最新事实；要求实时、专业或内部数据时，后续要通过受控工具或知识库提供来源。
- 任何请求都可能失败；错误信息应帮助用户处理问题，而不是暴露系统细节和密钥。

### 7.3 什么时候使用 `core/model_helper.py`

当前仓库的 `core/model_helper.py` 可集中管理聊天模型和 Embedding 的默认配置。完成本文件的 `hello_llm.py` 后，可以阅读并试用其中的：

- `get_llm_openai()`：返回 OpenAI 兼容客户端。
- `get_chat_openai()`：返回 LangChain 的 `ChatOpenAI` 对象。
- `get_embedding_model()`：返回 DashScope Embedding 对象，供 RAG 阶段使用。

复用前检查三项：模型名、`base_url` 和 `.env` 实际读取位置。配置文件的加载路径会影响从不同目录执行程序时能否读取到 Key；不要因为“代码里写了 `load_dotenv`”就跳过实际运行验证。

### Day 5 完成检查

- [ ] 已在 `notes.md` 记录至少一次成功请求和一次失败请求。
- [ ] 能说清楚模型 API、模型网页产品和本地 Python 程序之间的关系。
- [ ] 知道不能把模型输出直接作为 SQL、Shell 命令或文件路径执行。
- [ ] 能阅读 `core/model_helper.py` 的三个公开函数及其用途。

## 8. 常见问题排查表

| 现象 | 常见原因 | 检查顺序 |
| --- | --- | --- |
| 提示未找到 `QWEN_API_KEY` | `.env` 不在预期位置、变量名拼写错误、未调用 `load_dotenv()` | 确认运行目录 -> 查看变量名 -> 临时打印是否读取到布尔值，不打印 Key |
| 认证失败 | Key 无效、已过期、复制时带空格、账户无权限 | 在服务商控制台核验 -> 重新配置 -> 轮换 Key |
| 404 或模型不存在 | `base_url` 或模型名称不对，接口类型不兼容 | 对照当前服务商官方文档核验 URL 和模型名 |
| 连接失败或超时 | 网络、DNS、代理、公司网络限制 | 检查网络 -> 检查 `HTTPS_PROXY` -> 用最小请求重试 |
| 限流或余额不足 | 账户配额、并发或速率超限 | 查看服务商控制台 -> 降低频率 -> 加入退避重试 |
| 返回为空或格式异常 | 请求参数、模型行为、内容限制 | 打印非敏感响应元数据 -> 简化 Prompt -> 对输出做校验 |
| `ModuleNotFoundError` | 用错 Python 解释器或未安装依赖 | 使用 `uv run` -> 重新执行 `uv add` |

排错的原则是一次只验证一个假设。先运行最小调用程序，再逐步加入环境变量、封装、流式输出和框架；不要在报错时一次修改模型名、网络、依赖和代码。

## 9. 本阶段最终验收

在开始阶段 1 前，确保全部完成：

- [ ] 已理解 Token、上下文、模型、系统消息和用户消息的基本作用。
- [ ] 已使用 `uv` 创建隔离环境，并能用 `uv run` 运行脚本。
- [ ] `stage0/hello_llm.py` 能成功调用一个模型。
- [ ] API Key 仅存在于受忽略的本地配置或受控环境变量中。
- [ ] `stage0/.env.example` 只包含变量名和非敏感默认值。
- [ ] 能处理未配置 Key、认证失败和连接失败三种情况。
- [ ] 已保存一次成功和一次失败的学习记录，且不包含敏感信息。
- [ ] 已执行 `git status --short`，确认没有真实 Key、`.env` 或个人数据进入待提交文件。

## 10. 下一步

通过验收后，回到 [总路线的阶段 1](../llm-app-roadmap.md#3-阶段-1模型调用和-prompt-工程第-1-2-周)。下一阶段将基于这里的最小调用程序，学习消息历史、Prompt 设计和流式输出，并完成命令行 AI 文本助手。
