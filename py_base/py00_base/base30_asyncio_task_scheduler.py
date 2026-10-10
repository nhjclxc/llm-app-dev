

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Awaitable

import pytest

QUESTIONS = [
    {"id": 1, "question": "什么是 Python 协程？"},
    {"id": 2, "question": "asyncio.Queue 有什么作用？"},
    {"id": 3, "question": "如何限制异步任务并发数？"},
    {"id": 4, "question": "什么是 RAG？"},
    # {"id": 5, "question": "为什么需要设置请求超时？"},
    # {"id": 6, "question": "asyncio.create_task 和 gather 有什么区别？"},
    # {"id": 7, "question": "什么是大模型的 Token？"},
    # {"id": 8, "question": "如何处理 API 限流？"},
    # {"id": 9, "question": "什么是 Embedding？"},
    # {"id": 10, "question": "如何保存聊天历史？"},
    # {"id": 11, "question": "什么是函数调用？"},
    # {"id": 12, "question": "如何设计异步任务的错误处理？"},
]

@dataclass
class TaskResult:
    task_id: int
    question: str
    status: str
    answer: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    elapsed: float = 0.0


seed = 69

async def mock_llm_call(question: str, timeout: float) -> str:
    """模拟一次大模型 API 调用。"""

    try:
        async with asyncio.timeout(timeout):
            # random.seed(seed)
            rtime = random.uniform(0.2,3.0)
            if rtime > 2.0:
                raise RuntimeError("模拟模型服务临时失败")

            id = 1
            for ques in QUESTIONS:
                if question == ques.get("question"):
                    id = int(ques.get("id"))

            # 模拟处理问题
            await asyncio.sleep(rtime * id * 0.1)

            return f"answer: {question}"
    except asyncio.TimeoutError:
        raise RuntimeError("timeout")


async def process_one(
    task: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2,
    timeout_seconds: float = 2.0,
) -> TaskResult:
    """处理一个问题，包含并发限制、超时和重试。"""
    async with semaphore:
        id = task.get("id")
        question = task.get("question")
        # cancelled
        status = "success"
        answer = ""
        err_msg = ""
        attempt = 1
        delay = 0.5
        while attempt < max_retries+1:
            try:
                start = time.perf_counter()
                answer = await mock_llm_call(question, timeout_seconds)
                elapsed = time.perf_counter() - start
                return TaskResult(id, question, status, answer, err_msg, max_retries, elapsed)
            except RuntimeError as e:
                if e == "timeout":
                    status = "timeout"
            except Exception as e:
                err_msg = str(e)
                print(e)
                status = "failed"

            await asyncio.sleep(delay)
            attempt += 1
            delay += 0.5
        return TaskResult(id, question, status, answer, err_msg, max_retries)

async def run_with_gather(
    tasks: list[dict],
    concurrency: int = 3,
) -> list[TaskResult]:
    """使用 gather 并发处理全部任务。"""

    start = time.perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    async_tasks = [asyncio.create_task(process_one(question, semaphore, max_retries=2)) for question in tasks]
    res = await asyncio.gather(*async_tasks)
    # print(res)
    elapsed = time.perf_counter() - start
    print("elapsed", elapsed)
    return res


async def producer(
    queue: asyncio.Queue,
    tasks: list[dict[str, str]],
) -> None:
    """将任务放入队列。"""
    for task in tasks:
        await queue.put(task)
    await queue.put(None)
    pass


async def consumer(
    queue: asyncio.Queue,
    results: list[TaskResult],
    results_lock: asyncio.Lock,
    worker_id: int,
    semaphore: asyncio.Semaphore,
) -> None:
    """从队列获取任务并处理。"""
    while True:

        async with semaphore:
            task: dict[str, str] = queue.get()
            if task is None:
                break

            res = mock_llm_call(task.get("question"), timeout=1)
            async with results_lock:
                results.append(res)

    pass


async def run_with_queue(
    tasks: list[dict],
    worker_count: int = 3,
    queue_size: int = 5,
) -> list[TaskResult]:
    """使用生产者/消费者模型处理任务。"""
    pass


if __name__ == '__main__':

    pass