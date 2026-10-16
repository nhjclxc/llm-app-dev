

# python 的 asyncio 库必会的方法
# https://docs.python.org/zh-cn/3/library/asyncio.html

# asyncio 是 Python 的异步 I/O 框架
# async 定义协程，await 等待协程，create_task 创建并发任务，gather 批量等待任务，asyncio.run 启动整个异步程序。

"""

API	重要程度	用途
asyncio.run()	⭐⭐⭐⭐⭐	启动异步程序
async def	⭐⭐⭐⭐⭐	定义协程
await	⭐⭐⭐⭐⭐	等待异步操作
asyncio.create_task()	⭐⭐⭐⭐⭐	创建并发任务
asyncio.gather()	⭐⭐⭐⭐⭐	并发执行多个任务
asyncio.sleep()	⭐⭐⭐⭐⭐	异步休眠
asyncio.wait()	⭐⭐⭐⭐	等待任务集合
asyncio.wait_for()	⭐⭐⭐⭐	给协程设置超时
asyncio.timeout()	⭐⭐⭐⭐	异步超时上下文
asyncio.Queue	⭐⭐⭐⭐⭐	异步生产者/消费者
asyncio.Lock	⭐⭐⭐⭐	异步锁
asyncio.Semaphore	⭐⭐⭐⭐⭐	限制并发数量
asyncio.Event	⭐⭐⭐	异步事件通知
asyncio.Future	⭐⭐⭐	底层异步结果对象
asyncio.to_thread()	⭐⭐⭐⭐⭐	在线程中执行同步函数

"""

import asyncio
import time

import aiohttp


async def main():
    print('1 ...')
    # asyncio.sleep(1) 异步休眠表示的意思不是：阻塞线程 1 秒。
    # 而是：当前协程暂时没有事情做，把执行权交给事件循环。即让出cpu执行权限给其他事件，使得cpu不忙等，只有在多协程里面才有体现
    await asyncio.sleep(1)
    # time.sleep(1) 会阻塞当前线程。那么这个线程什么都干不了。
    # await asyncio.sleep(1)只暂停当前协程。事件循环可以去执行其他任务：
    print('... 2!')


async def main2():
    await main()

async def work(name: str, loop:int):
    print('{} work ...'.format(name))
    await asyncio.sleep(loop)
    print('{} work done'.format(name))

async def func_async2():

    # asyncio.create_task() 创建事件循环任务
    # A, B, C 三个任务会在统一时刻创建
    # 同时创建三个 task 任务
    task1 = asyncio.create_task(work('A', 3))
    task2 = asyncio.create_task(work('B', 2))
    task3 = asyncio.create_task(work('C', 1))

    # 同时等待执行完毕， 多个任务要同时创建同时等待才是并发执行
    await task1
    await task2
    await task3

    pass


async def func_fetch(url: str, timeout:int):
    # 被 async 修饰的函数叫做 awaitable 可等待的函数
    print('{} fetch, timeout {} ...'.format(url, timeout))
    await asyncio.sleep(timeout)
    print('{} fetch done'.format(url))
    return f" {url} -> {timeout} "

async def func_gather():
    # asyncio.gather 将多个任务合并一起创建一起执行一起等待
    # 多个协程
    #    ↓
    # gather
    #    ↓
    # 并发执行
    #    ↓
    # 等待全部完成
    # asyncio.gather 可以获取异步协程的返回值
    res = await asyncio.gather(func_fetch('A', 3), func_fetch('B', 2), func_fetch('C', 1))
    print(res)

    # create_task 重点：把一个协程包装成 Task，并立即安排到事件循环执行。
    # gather 重点： 把多个 awaitable 聚合起来，同时运行并等待结果。

    pass


async def func_wait():
    # asyncio.wait() 等待一批任务的执行完毕

    tasks:list[asyncio.Task] = []
    tasks.append(asyncio.create_task(work('A', 3)))
    tasks.append(asyncio.create_task(work('B', 2)))
    tasks.append(asyncio.create_task(work('C', 1)))

    # 使用 asyncio.wait 等待一批任务的执行完毕
    #  wait 比gather 更底层
    done, pending = await asyncio.wait(tasks)
    print(done)
    print()
    print(pending)


    # wait_for 给异步任务设置超时时间
    try:
        result = await asyncio.wait_for(
            func_fetch('C', 3),
            timeout=2,
        )
        print("result: ", result)
    except asyncio.TimeoutError:
        print("timeout")

    pass


async def func_timeout():
    # asyncio.timeout(3): 等待超过某个时间后开始执行

    try:
        # asyncio.timeout(3) 使用把代码放到超时时间内，如果超过某个时间没有返回就触发超时异常
        async with asyncio.timeout(3):
            print('start')
            res = await func_fetch('C', 5)
            print("res: ", res)
    except asyncio.TimeoutError:
        print('timeout')

    pass


async def func_queue_producer(q: asyncio.Queue):
    # put 是异步函数
    await q.put("now")
    print('queue producer')
    for i in range(3):
        await asyncio.sleep(1)
        await q.put(i)
        print(f"{i} put done")
    await q.put(None)
    return "producer done"

async def func_queue_consumer(q: asyncio.Queue):
    res = await q.get()
    print("res: ", res)
    # 消息处理完毕回调
    q.task_done()

    while True:
        try:
            res = await q.get()
            if res is None:
                print("rev done flag")
                break
            print("res: ", res)
        finally:
            q.task_done()
    return 'consume done'

async def func_queue():
    # asyncio.Queue 循环事件中进行消息传递

    queue = asyncio.Queue()
    res = await asyncio.gather(func_queue_producer(queue), func_queue_consumer(queue))
    print(res)

    # 等待队列被清空
    await queue.join()
    print("queue clear up ")

    pass

async def queue2_producer(q: asyncio.Queue):
    for i in range(10):
        # 生产速度必须大于消费速度的时候才要进行多消费者
        await asyncio.sleep(0.1)
        url = f"http://example.org/page{i}"
        print(f"producer -> {url}")
        await q.put(url)

    for _ in range(3):
        await q.put(None)

    pass

async def queue2_consumer(q: asyncio.Queue, workerId: int):
    while True:
        try:
            res = await q.get()
            if res is None:
                print(f"workerId: {workerId} -> {res} done")
                break

            await asyncio.sleep(1)
            print(f"workerId: {workerId} -> {res}")
        finally:
            q.task_done()

async def func_queue2():
    # 单生产者-多消费者
    queue = asyncio.Queue()

    producer = asyncio.create_task(queue2_producer(queue))

    consumers = [
        asyncio.create_task(queue2_consumer(queue, i+1)) for i in range(3)
    ]

    await producer

    await queue.join()

    for task in consumers:
        task.cancel()

    pass

async def func_fetch_url(sem: asyncio.Semaphore ,url: str):
    # 控制同时接入临界区的worker数量
    # sem 对象必须在协程里面使用，这样每个协程进来的时候才能控制
    async with sem:
        print('fetch url', url)
        # 模拟任务执行
        await asyncio.sleep(1)

async def func_semaphore():
    #  asyncio.Semaphore
    # 最多允许 3 个任务同时进入临界区域。
    # 使用 Semaphore 控制并发数
    sem = asyncio.Semaphore(3)
    tasks = [
        asyncio.create_task(func_fetch_url(sem, f"http://example.org/page{i}"))
        for i in range(10)
    ]
    # *tasks 表示等待里面所有的task
    await asyncio.gather(*tasks)

    pass

global_data = 0

async def func_lock_work(lock: asyncio.Lock, workerId: int ):
    global global_data
    # 锁共享区变量
    async with lock:
        print(f"lock work {workerId} : {global_data} before")
        global_data = workerId
        print(f"lock work {workerId} : {global_data} after")

async def func_lock():
    # asyncio.Lock 异步锁 防止共享区数据同时被多个协程修改
    lock = asyncio.Lock()

    tasks = [
        asyncio.create_task(func_lock_work(lock, i))
        for i in range(10)
    ]
    await asyncio.gather(*tasks)

    print("done ")
    pass

async def func_event_set(event: asyncio.Event):
    print('event set')
    await asyncio.sleep(1)
    event.set()
    print('event set done')
    pass
async def func_event_wait(event: asyncio.Event):
    print('event wait')
    await event.wait()
    print('event wait done')
    event.clear()
    pass

async def func_event():
    # asyncio.Event
    # 一个协程通知其他协程某个事件发生了。
    # 等待 wait，触发 set，清除 clear

    event = asyncio.Event()

    task1 = asyncio.create_task(func_event_wait(event))
    task2 = asyncio.create_task(func_event_set(event))

    await asyncio.wait([task1, task2])

    print('event done')

    pass

def func_block_work():
    time.sleep(3)

async def func_block_heartbeat():
    for i in range(10):
        print("heartbeat", i, time.time())
        await asyncio.sleep(0.5)

async def func_to_thread():
    # asyncio.to_thread()
    # asyncio 只能很好地处理异步 I/O，不能自动把同步阻塞代码(如time.sleep/io等)变成异步。
    # 那么此时就可以使用 asyncio.to_thread 来包裹同步阻塞代码

    task = asyncio.create_task(func_block_heartbeat())
    # func_to_thread start 1790231013.689342
    # func_to_thread end  1790231016.6943989
    # 这里 func_block_work 被阻塞了，阻塞了3s没有输出heartbeat
    print('func_to_thread start', time.time())
    func_block_work()
    print('func_to_thread end ', time.time())
    await task

    task2 = asyncio.create_task(func_block_heartbeat())
    # func_to_thread2 start 1790231024.711162
    # func_to_thread2 end  1790231027.711214
    # 这里没有直接阻塞asyncio.to_thread(func_block_work)，没有阻塞，会不断输出heartbeat
    print('func_to_thread2 start', time.time())
    await asyncio.to_thread(func_block_work)
    print('func_to_thread2 end ', time.time())
    await task2
    pass

if __name__ == '__main__':

    # 使用 async 定义的函数，如果直接使用main()返回的是一个 coroutine 对象，此时还没有执行该函数
    # coro = main()
    # print('coro:', coro, ", type: ", type(coro))

    # 使用 asyncio.run 来运行这个 coroutine 对象，启动一个协程
    # 或者使用 await main() 来运行，使用 await 的时候当前函数也必须是一个async定义的函数
    # asyncio.run 里面创建出来的函数就叫：事件循环
    # asyncio.run(main())

    # asyncio.run(func_async2())
    # asyncio.run(func_gather())
    # asyncio.run(func_wait())
    # asyncio.run(func_timeout())
    # asyncio.run(func_queue())
    # asyncio.run(func_queue2())
    # asyncio.run(func_semaphore())
    # asyncio.run(func_lock())
    # asyncio.run(func_event())
    asyncio.run(func_to_thread())




    pass



