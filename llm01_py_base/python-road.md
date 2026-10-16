


[马士兵教育](https://www.processon.com/view/link/677f8c5638625f07c0fd0cc3)



如果你的目标是真正做大模型应用开发 / Agent 开发，而不是研究模型训练，我建议按下面这张技术栈清单学习：

Python —— 大模型应用开发的主力语言，重点掌握 asyncio、httpx、typing、pydantic 等。
HTTP / REST API —— 理解如何调用 LLM、Embedding、Rerank、业务服务。
OpenAI API —— 学会 Chat、Structured Output、Tool Calling、Streaming、Embeddings 等核心能力。
Anthropic API —— 另一套主流 LLM API，重点理解 Claude 的 Tool Use、Context 等机制。
Pydantic —— 用 Python 类型定义和校验 LLM 的结构化输入输出。
Jinja2 —— Prompt 模板化。
Prompt Engineering —— System Prompt、Few-shot、CoT、结构化输出、Prompt 模板设计。
Streaming / SSE —— 实现 ChatGPT 类似的流式输出。
Function Calling / Tool Calling —— 让大模型调用 Python 函数、HTTP API、数据库等工具。
MCP（Model Context Protocol） —— 让模型以统一协议连接工具、数据库、文件系统、第三方服务。
LangChain —— LLM 应用基础框架，封装 Model、Prompt、Tool、Retriever、Agent 等。
LangGraph —— 构建复杂 Agent 工作流，支持 State、节点、条件分支、循环、人工介入。
LlamaIndex —— 更偏 RAG / 数据接入 / 文档索引的大模型应用框架。
Hermes —— 主要用于 Agent / Tool Calling 相关的模型与工具交互生态，建议在掌握 MCP、Tool Calling 后学习。
RAG（Retrieval-Augmented Generation） —— 让模型查询企业文档、知识库后再回答。
Embedding —— 将文本转换成向量，用于语义检索。
Vector Database —— 向量数据库，如 Milvus、Qdrant、Weaviate、pgvector。
Reranker —— 对向量检索结果重新排序，提高 RAG 检索准确率。
Chunking —— 文档切片，决定 RAG 能否正确检索知识。
Hybrid Search —— 关键词检索 + 向量检索结合。
Elasticsearch / OpenSearch —— 全文搜索、关键词搜索、混合搜索的重要基础设施。
Redis —— Session、缓存、Agent 状态、限流、队列等。
PostgreSQL —— 用户、业务数据、Agent 数据、RAG Metadata 等持久化存储。
SQLAlchemy —— Python 数据库 ORM / 数据访问层。
FastAPI —— 把你的 LLM / Agent 做成 HTTP API 服务。
Celery / Redis Queue —— 执行异步、延迟、后台任务。
Kafka —— 大规模 Agent / LLM 应用中的异步事件流和数据管道。
Docker —— 打包 LLM 应用、Agent、向量库等服务。
Kubernetes —— 大规模部署和管理 AI 应用。
vLLM —— 自部署开源大模型的高性能推理服务。
Ollama —— 本地运行开源大模型，适合开发和实验。
Hugging Face Transformers —— 使用开源模型的核心 Python 框架。
Model Context / Context Window —— 理解模型上下文长度、Token 消耗、上下文压缩。
Tokenization —— 理解 Token、Token 数量和上下文成本。
Context Engineering —— 如何把正确的信息、工具、历史对话放进模型上下文。
Memory —— Agent 的短期记忆、长期记忆、用户记忆。
Agent —— 模型自主决定下一步行动、调用工具、观察结果并继续执行。
Agent Workflow —— 将 Agent 拆成 State → Node → Tool → Condition → Loop 等流程。
Multi-Agent —— 多个 Agent 协作完成复杂任务。
ReAct —— Reasoning + Acting 的经典 Agent 思路。
Structured Output —— 强制模型输出符合 JSON Schema / Pydantic 的结构。
Guardrails —— 对模型输入输出进行格式、安全、业务规则校验。
LLM Evaluation —— 评估模型回答质量、准确率、召回率、Faithfulness 等。
Ragas —— RAG 评测框架。
DeepEval —— LLM / Agent 自动化评测框架。
LangSmith —— LangChain / Agent 的 Trace、Debug、Evaluation、Monitoring。
OpenTelemetry —— LLM / Agent 服务的标准化 Trace、Metrics、Logs。
Prometheus —— 监控 QPS、延迟、Token、错误率等业务指标。
Grafana —— 可视化 LLM / Agent 运行指标。
LLM Gateway —— 统一管理 OpenAI、Claude、Gemini、国产模型等多个模型 API。
LiteLLM —— 一个很常用的多模型统一调用层。
AI SDK —— 简化前后端 AI 应用、Streaming、Tool Calling 等开发。
Gradio —— 快速制作 AI Demo / 内部工具 UI。
Streamlit —— 快速制作数据 / AI 应用 Web UI。
Next.js / React —— 如果自己开发正式的 AI Web 产品，需要掌握前端。
OAuth 2.0 / JWT —— AI 应用用户认证和权限控制。
Rate Limiting —— 控制 LLM API / Agent 的请求频率和成本。
Prompt Cache / Semantic Cache —— 减少重复 LLM 请求和 Token 成本。
Batch / Async / Concurrent —— 大规模 LLM 请求的并发和成本优化。
AI Security —— Prompt Injection、Data Leakage、Tool Abuse、越权调用等安全问题。

