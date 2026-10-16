# asyncio 综合练习题：异步批量任务调度器

## 1. 练习背景

你正在开发一个大模型应用，需要批量处理一组用户问题。每个问题都要经过一个“模型接口”处理，接口可能出现以下情况：

- 正常返回结果；
- 响应时间较长；
- 临时失败，需要重试；
- 超过最大等待时间；
- 同时请求过多，需要限制并发数。

本练习要求你使用 Python 标准库 `asyncio`，实现一个本地异步任务调度器。为了避免依赖真实网络和 API Key，模型接口使用异步函数模拟。

你需要完成：任务生产、并发执行、超时控制、失败重试、并发限制、结果收集、任务取消和统计汇总。

## 2. 练习目标

完成本题后，你应该能够理解并使用：

- `async def` 定义协程；
- `await` 等待异步操作；
- `asyncio.run()` 启动事件循环；
- `asyncio.create_task()` 创建任务；
- `asyncio.gather()` 等待多个任务；
- `asyncio.wait()` 获取已完成和未完成任务；
- `asyncio.timeout()` 或 `asyncio.wait_for()` 设置超时；
- `asyncio.sleep()` 模拟异步 I/O 等待；
- `asyncio.Semaphore` 限制并发量；
- `asyncio.Queue` 实现生产者/消费者模型；
- `asyncio.Lock` 保护共享统计数据；
- `asyncio.Event` 广播停止信号；
- `asyncio.CancelledError` 处理任务取消；
- `asyncio.to_thread()` 将阻塞函数放到线程中执行；
- `dataclasses` 或类型标注组织任务结果；
- 日志、异常和统计信息的基本设计。

## 3. 文件要求

请在 `py_base/py00_base/` 下新建：

```text
base30_asyncio_task_scheduler.py
```

建议最终结构：

```text
py_base/py00_base/
  base29_asyncio.py
  base30_asyncio_task_scheduler.py
```

本题只能使用 Python 标准库，不允许使用 `aiohttp`、`httpx`、`requests` 等第三方网络库。

允许使用：

```python
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional
```

## 4. 模拟数据

准备至少 12 条待处理问题，每条数据包含唯一 ID 和问题文本：

```python
QUESTIONS = [
    {"id": 1, "question": "什么是 Python 协程？"},
    {"id": 2, "question": "asyncio.Queue 有什么作用？"},
    {"id": 3, "question": "如何限制异步任务并发数？"},
    {"id": 4, "question": "什么是 RAG？"},
    {"id": 5, "question": "为什么需要设置请求超时？"},
    {"id": 6, "question": "asyncio.create_task 和 gather 有什么区别？"},
    {"id": 7, "question": "什么是大模型的 Token？"},
    {"id": 8, "question": "如何处理 API 限流？"},
    {"id": 9, "question": "什么是 Embedding？"},
    {"id": 10, "question": "如何保存聊天历史？"},
    {"id": 11, "question": "什么是函数调用？"},
    {"id": 12, "question": "如何设计异步任务的错误处理？"},
]
```

## 5. 第一部分：实现模拟模型接口

实现以下协程：

```python
async def mock_llm_call(question: str, attempt: int = 1) -> str:
    """模拟一次大模型 API 调用。"""
```

要求：

1. 使用 `await asyncio.sleep(...)` 模拟网络等待。
2. 每次调用的等待时间为随机值，例如 `0.2` 到 `3.0` 秒。
3. 根据一定概率抛出临时异常，例如：

   ```python
   raise RuntimeError("模拟模型服务临时失败")
   ```

4. 成功时返回：

   ```text
   已处理问题：{question}
   ```

5. 函数中禁止使用 `time.sleep()`，否则会阻塞整个事件循环。
6. 为了便于测试，建议支持固定随机种子或通过参数传入模拟行为。

可选增强：让不同问题拥有不同的响应耗时和失败概率，而不是完全随机。

## 6. 第二部分：定义任务结果对象

使用 `dataclass` 定义 `TaskResult`：

```python
@dataclass
class TaskResult:
    task_id: int
    question: str
    status: str
    answer: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    elapsed: float = 0.0
```

`status` 至少支持以下值：

```text
success
failed
timeout
cancelled
```

要求：

- 成功时填写 `answer`、`attempts` 和 `elapsed`；
- 失败时填写 `error`；
- 超时时不能被错误地记录为普通失败；
- 被取消时不能被错误地记录为成功；
- `elapsed` 使用 `time.perf_counter()` 计算，不要使用系统时间差代替耗时统计。

## 7. 第三部分：实现单个任务处理函数

实现：

```python
async def process_one(
    task: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2,
    timeout_seconds: float = 2.0,
) -> TaskResult:
    """处理一个问题，包含并发限制、超时和重试。"""
```

### 7.1 并发限制

使用 `asyncio.Semaphore`，要求同时执行的模型调用不超过指定数量，例如最多 3 个：

```python
async with semaphore:
    # 在这里执行一次模型调用
```

要求：

- Semaphore 只保护真正的模型调用部分；
- 不要因为一个任务等待重试，就长时间占用 Semaphore；
- 代码中不能直接通过 `time.sleep()` 等待。

### 7.2 重试规则

对临时失败进行重试：

- 第一次失败后等待 `0.5` 秒；
- 第二次失败后等待 `1.0` 秒；
- 后续等待时间可以继续递增，但不能无限重试；
- 最大尝试次数为 `max_retries + 1`；
- 最后一次仍失败时返回 `failed`。

重试等待必须使用：

```python
await asyncio.sleep(delay)
```

不要捕获所有异常后无限重试。至少区分：

- 模拟服务临时错误：可以重试；
- 超时：根据需求决定是否重试，但要有明确上限；
- `asyncio.CancelledError`：必须继续向上抛出或单独处理，不能把取消吞掉后继续执行。

### 7.3 超时规则

每次调用都必须有超时限制：

```python
async with asyncio.timeout(timeout_seconds):
    answer = await mock_llm_call(question, attempt)
```

如果使用 Python 3.10 或更低版本，可以改用：

```python
answer = await asyncio.wait_for(
    mock_llm_call(question, attempt),
    timeout=timeout_seconds,
)
```

超时后：

- 返回 `status="timeout"`；
- 保存清晰的错误信息；
- 不得让一个超时任务阻塞其他任务；
- 是否重试必须由你的设计明确决定，建议超时最多重试一次。

## 8. 第四部分：使用 `asyncio.gather` 实现批量处理

实现：

```python
async def run_with_gather(
    tasks: list[dict],
    concurrency: int = 3,
) -> list[TaskResult]:
    """使用 gather 并发处理全部任务。"""
```

要求：

1. 创建一个 `asyncio.Semaphore(concurrency)`。
2. 为每条任务创建一个协程或 Task。
3. 使用 `asyncio.gather(...)` 等待所有任务完成。
4. 返回结果列表，结果顺序必须与输入任务顺序一致。
5. 单个任务失败不能导致其他任务结果全部丢失。
6. 在函数开始和结束位置记录总耗时。

### 8.1 必须回答的问题

在代码注释或 `notes.md` 中回答：

- 为什么先创建多个任务，再统一 `gather`，能够并发执行？
- 如果依次写 `await process_one(...)`，为什么会退化为串行？
- `gather` 默认遇到异常时会怎样？本题为什么建议让 `process_one` 自己返回 `TaskResult`？
- `return_exceptions=True` 适合什么场景？它有什么代价？

## 9. 第五部分：使用 `asyncio.Queue` 改造成生产者/消费者模型

实现：

```python
async def producer(
    queue: asyncio.Queue,
    tasks: list[dict],
) -> None:
    """将任务放入队列。"""


async def consumer(
    queue: asyncio.Queue,
    results: list[TaskResult],
    results_lock: asyncio.Lock,
    worker_id: int,
    semaphore: asyncio.Semaphore,
) -> None:
    """从队列获取任务并处理。"""


async def run_with_queue(
    tasks: list[dict],
    worker_count: int = 3,
    queue_size: int = 5,
) -> list[TaskResult]:
    """使用生产者/消费者模型处理任务。"""
```

### 9.1 队列要求

- 创建有界队列：`asyncio.Queue(maxsize=queue_size)`；
- 生产者使用 `await queue.put(task)`；
- 消费者使用 `task = await queue.get()`；
- 每个正常取出的任务必须最终调用 `queue.task_done()`；
- 主协程使用 `await queue.join()` 等待所有任务处理完成；
- 使用特殊结束标记，例如 `None`，通知消费者退出；
- 结束标记数量必须与消费者数量匹配，或者使用统一的停止事件方案。

### 9.2 结果共享要求

多个消费者会同时写入 `results`。要求使用 `asyncio.Lock` 保护共享数据：

```python
async with results_lock:
    results.append(result)
```

要求解释：

- 为什么普通的 `list.append()` 看起来安全，但仍建议在共享状态边界使用锁？
- `asyncio.Lock` 和线程锁的适用范围有什么不同？
- 锁的临界区为什么应该尽量短？

### 9.3 队列模型与 gather 模型对比

在 `notes.md` 中比较：

| 对比项 | `gather` | `Queue` + worker |
| --- | --- | --- |
| 适合任务数量 | 少量、已知任务 | 大量或持续产生的任务 |
| 结果顺序 | 默认保持输入顺序 | 通常按完成顺序收集 |
| 背压 | 不明显 | 有界队列可以形成背压 |
| worker 数量 | 由任务数量决定 | 可固定 worker 数量 |
| 任务生产和消费 | 耦合在一起 | 可以解耦 |

## 10. 第六部分：增加取消和优雅停止

实现：

```python
async def cancel_demo(tasks: list[asyncio.Task]) -> None:
    """演示取消尚未完成的任务。"""
```

要求：

1. 启动批量任务。
2. 等待一小段时间，例如 `0.8` 秒。
3. 取消仍未完成的任务：

   ```python
   task.cancel()
   ```

4. 使用 `asyncio.gather(*tasks, return_exceptions=True)` 等待取消完成。
5. 识别 `asyncio.CancelledError`，将结果记录为 `cancelled` 或输出取消统计。
6. 不能留下未回收的 Pending Task。

生产者/消费者版本还要增加：

- `asyncio.Event` 作为停止信号；
- 主协程发生异常时，通知消费者停止接收新任务；
- 退出前等待或取消所有 worker；
- 使用 `try/finally` 确保 `queue.task_done()` 被调用。

## 11. 第七部分：实现同步阻塞函数的异步适配

实现一个故意阻塞的函数：

```python
def blocking_report(task_id: int) -> str:
    time.sleep(1)
    return f"task {task_id} report generated"
```

要求：

1. 先直接在协程中调用它，观察事件循环被阻塞的现象。
2. 再使用：

   ```python
   result = await asyncio.to_thread(blocking_report, task_id)
   ```

3. 对比两种写法的总耗时。
4. 在注释中说明：

   - `asyncio.to_thread()` 适合什么类型的任务；
   - 它不是把 CPU 密集型计算 magically 变成异步；
   - 真正的 CPU 密集型任务通常需要进程池或其他方案；
   - 阻塞的文件、旧式同步 SDK 和同步数据库调用可能适合放到线程中。

## 12. 第八部分：统计和输出报告

实现汇总函数：

```python
def build_report(results: list[TaskResult], total_elapsed: float) -> str:
    """生成任务处理报告。"""
```

报告至少包含：

```text
总任务数：12
成功数：...
失败数：...
超时数：...
取消数：...
总耗时：...
平均单任务耗时：...
最大单任务耗时：...
总尝试次数：...
```

还要输出每条任务的简要结果：

```text
[success] task_id=1 attempts=1 elapsed=0.42s
[timeout] task_id=2 attempts=2 elapsed=4.05s error=...
```

要求：

- 结果按 `task_id` 排序后输出；
- 报告中不得输出 API Key 或其他敏感信息；
- 报告中能区分“模型返回失败”和“本地超时”；
- 每次运行都能通过固定随机种子复现实验结果，或说明随机结果为什么会变化。

## 13. 主程序要求

实现：

```python
async def main() -> None:
    """依次运行并比较不同调度方式。"""
```

主程序至少执行以下实验：

### 实验 A：串行版本

- 依次 `await` 每个任务；
- 记录总耗时。

### 实验 B：`gather` 版本

- 并发创建全部任务；
- 使用 Semaphore 限制并发数为 3；
- 记录总耗时。

### 实验 C：Queue 版本

- 生产者将任务放入有界队列；
- 启动 3 个消费者；
- 使用 `queue.join()` 等待完成；
- 记录总耗时。

### 实验 D：取消版本

- 启动任务后提前取消；
- 输出取消数量；
- 确认程序可以正常退出。

### 实验 E：同步阻塞对比

- 对比直接执行 `blocking_report()` 和 `asyncio.to_thread()` 的行为。

程序入口必须使用：

```python
if __name__ == "__main__":
    asyncio.run(main())
```

## 14. 功能验收标准

### 14.1 必做验收

- [ ] 程序可以使用 `asyncio.run(main())` 启动。
- [ ] 至少有 12 个模拟任务。
- [ ] 模拟接口使用 `await asyncio.sleep()`，没有在协程中使用 `time.sleep()`。
- [ ] 能限制同时执行的模型调用数量。
- [ ] 每个任务都有超时处理。
- [ ] 临时失败最多重试指定次数。
- [ ] 结果包含成功、失败、超时和取消状态。
- [ ] `gather` 版本不会因为一个任务失败而丢失其他任务结果。
- [ ] Queue 版本正确调用 `queue.task_done()` 和 `queue.join()`。
- [ ] Queue 版本可以正常停止所有消费者。
- [ ] 共享结果列表使用 `asyncio.Lock` 保护。
- [ ] 取消任务后没有 Pending Task 或未处理异常。
- [ ] 能输出汇总统计报告。
- [ ] 已对比串行、gather 和 Queue 三种方案的耗时。

### 14.2 代码质量验收

- [ ] 函数有明确的参数和返回值类型标注。
- [ ] 协程函数、生产者、消费者和统计逻辑职责清晰。
- [ ] 不使用全局可变变量保存任务结果。
- [ ] 不使用无限重试或无限等待。
- [ ] 对 `asyncio.CancelledError` 不做吞异常处理。
- [ ] 异常信息清晰，包含任务 ID。
- [ ] 代码中没有真实 API Key、Cookie 或私有数据。
- [ ] 关键状态变更有日志或可读的终端输出。

## 15. 思考题

请将以下问题的答案写入同目录的 `asyncio思考题.md`：

1. `async def` 函数调用后为什么不会立即执行函数体？
2. `await` 的作用是什么？它是否会创建并发任务？
3. `create_task()` 和直接 `await` 的区别是什么？
4. 为什么 `asyncio.sleep()` 不阻塞事件循环，而 `time.sleep()` 会阻塞？
5. `asyncio.gather()` 和 `asyncio.wait()` 如何选择？
6. `asyncio.timeout()` 超时后，内部任务发生了什么？
7. 为什么一个协程中不应该捕获 `CancelledError` 后直接忽略？
8. Semaphore、Lock 和 Queue 分别解决什么问题？
9. 为什么有界 Queue 能够产生背压？
10. 什么时候应该使用 `asyncio.to_thread()`？
11. 为什么“协程”不等于“多线程”？
12. 如果处理 100 万条任务，直接创建 100 万个 Task 有什么风险？
13. 如何在生产环境中实现全局并发限制、每个用户并发限制和每个模型并发限制？
14. 如果任务具有写数据库、扣费或发送消息等副作用，重试时需要注意什么？
15. 如果一个消费者异常退出，如何避免队列永远无法 `join()`？

## 16. 进阶要求

完成必做部分后，再实现以下增强。

### 16.1 任务优先级

将 `asyncio.Queue` 替换为 `asyncio.PriorityQueue`，让高优先级问题优先处理。

要求：

- 优先级数值越小，优先级越高；
- 相同优先级仍能稳定处理；
- 结果输出仍按任务 ID 排序，而不是按优先级排序。

### 16.2 动态限流

根据模拟服务返回的“限流错误”动态降低并发数，并在一段时间后逐步恢复。

要求：

- 不能简单地无条件降低到 1；
- 限流后使用退避等待；
- 记录并发数变化和限流次数。

### 16.3 异步生成器

实现一个异步生成器，任务完成一个就立即产出一个结果：

```python
async def iter_completed_results(tasks):
    yield result
```

要求：

- 结果按完成顺序产生；
- 主程序可以边处理边打印；
- 不能等所有任务完成后才一次性返回。

### 16.4 异步上下文管理器

实现一个简单的 `AsyncTaskRunner`：

```python
async with AsyncTaskRunner(concurrency=3) as runner:
    result = await runner.run(task)
```

要求实现 `__aenter__` 和 `__aexit__`，并在退出时取消和回收未完成任务。

### 16.5 接入真实 HTTP API

完成本题全部本地模拟后，才允许将 `mock_llm_call()` 替换为真实异步 HTTP 调用。

要求：

- API Key 从环境变量读取；
- 不得写入源码；
- 设置连接超时和读取超时；
- 限制并发；
- 对 429、5xx、连接错误和超时进行分类处理；
- 日志中不得输出完整请求头和 API Key。

## 17. 运行和验证建议

在仓库根目录执行：

```bash
python py_base/py00_base/base30_asyncio_task_scheduler.py
```

建议固定随机种子后至少运行 3 次，分别验证：

1. 全部成功的情况；
2. 存在临时失败并成功重试的情况；
3. 存在最终失败和超时的情况；
4. 提前取消任务的情况；
5. Queue 消费者正常退出的情况。

如果使用 Python 3.10 或更低版本，不能使用 `asyncio.timeout()`，应使用 `asyncio.wait_for()` 替代，并在 `notes.md` 中记录版本差异。

## 18. 最终提交内容

完成后，`py_base/py00_base/` 下至少应包含：

```text
base30_asyncio_task_scheduler.py
asyncio思考题.md
```

代码文件顶部写明：

```python
"""asyncio 综合练习：异步批量任务调度器。"""
```

同时在代码注释或 `asyncio思考题.md` 中记录：

- 串行、gather、Queue 三种方案的耗时；
- 你设置的并发数、超时时间和重试次数；
- 一个失败任务的完整状态变化；
- 一个取消任务的处理过程；
- 你认为生产环境还需要补充的两个可靠性措施。

