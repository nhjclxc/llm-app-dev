

# python 的 queue 库必会的方法
# https://docs.python.org/zh-cn/3/library/queue.html



import queue
import threading
import time

"""


queue
  ↓
线程之间传递任务 / 数据
  ↓
生产者 ──put()──→ Queue ──get()──→ 消费者

它最典型的场景就是：生产者线程不断产生任务，消费者线程不断取任务处理。



⭐⭐⭐⭐⭐
Queue
put()
get()
task_done()
join()

⭐⭐⭐⭐
empty()
full()
qsize()

⭐⭐⭐
put_nowait()
get_nowait()

PriorityQueue
LifoQueue

⭐⭐
SimpleQueue

"""

def queue_work(q: queue.Queue):
    time.sleep(3)
    print("Queue worker started, ", q.get())

def func_queue():
    # 使用 maxsize 来形成一种背压
    q = queue.Queue(maxsize=3)
    # 默认先进先出

    q.put("任务1")
    q.put("任务2")
    q.put("任务3")

    # get 默认阻塞，也支持block，timeout
    print(q.get())
    print(q.get())
    print(q.get())

    q.put("任务1")
    q.put("任务2")
    q.put("任务3")
    threading.Thread(target=queue_work, args=(q,)).start()

    print("投递 任务333")
    # block=True表示阻塞等待
    # timeout=5表示阻塞等待最大时长为5s
    q.put("任务333", block=True, timeout=5)
    # put_nowait 不等待，放不进去马上返回/抛异常。
    # q.put_nowait(123)
    print("任务333 投递完成 ")

    # 当get()获取到一个队列数据处理完成之后必须使用task_done(),因为queue内部会维护一个未完成任务数量来进行计数
    data = q.get()
    try:
        time.sleep(1)
        print("处理完成： ", data)
    finally:
        q.task_done()
        print("任务标记已完成")

    pass

def worker_queue_join(q: queue.Queue):
    while True:
        data = q.get()
        try:
            print("处理：", data)
            time.sleep(1)
        finally:
            q.task_done()
            print("标记处理完成：", data)

def func_queue_join():

    # queue.Queue.join() 等待 Queue 中所有任务都处理完成。
    # join() + task_done() 是一对

    q = queue.Queue(maxsize=3)

    threading.Thread(target=worker_queue_join, args=(q,), daemon=True).start()

    # 投递数据
    for data in range(10):
        time.sleep(0.5)
        q.put(data)
        print("数据投递完成：", data)

    print("等待队列清空")
    q.join()
    print("队列清空完成")


    pass


# 使用 queue 实现生产者-消费者模式
def func_producer(q: queue.Queue):
    for i in range(5):
        time.sleep(0.5)
        q.put(i)
        print("数据投递：", i)
        print("队列大小：", q.qsize())

    # 使用 哨兵值方法来告诉消费者可以推出
    # 生产者已经结束了，你可以退出了。
    q.put(None)

    print("生产者数据投递完毕")

def func_consumer(q: queue.Queue, index: int = -1):
    total = 0
    while True:
        # # 消费方法1： 获取任务和处理任务后合并在一起
        # consumer_flag = False
        # try:
        #     data = q.get(timeout=3)
        #     print("消费数据完成：", data)
        #     total += 1
        #     consumer_flag = True
        # except queue.Empty:
        #     print("读取数据失败")
        #     break
        # finally:
        #     if consumer_flag:
        #         q.task_done()
        #
        # # 消费方法2: 获取任务和处理任务分开
        #
        # try:
        #     data = q.get(timeout=3)
        # except queue.Empty:
        #     print("3秒没有数据，消费者退出")
        #     break
        #
        # try:
        #     print("消费数据完成：", data)
        #     total += 1
        # finally:
        #     q.task_done()

        # 消费方法3: 哨兵值
        try:
            data = q.get()
            # 读取到哨兵值，直接退出循环
            if data is None:
                break
            print(f"消费数据完成 {index}：", data)
            total += 1
        finally:
            q.task_done()

    print("所有数据消费完毕 total：", total)

def func_producer_consumer():

    q = queue.Queue(maxsize=3)

    consumer_thread = threading.Thread(target=func_consumer, args=(q,), daemon=True)
    consumer_thread.start()
    producer_thread = threading.Thread(target=func_producer, args=(q,))
    producer_thread.start()

    # 等待完成
    consumer_thread.join()

    # 等待队列被清空
    q.join()

    print("消费完成")

    pass


def func_producer_mult_consumer():
    # 单生产者-多消费者

    q = queue.Queue(maxsize=3)

    consumer_threads = []
    for i in range(3):
        consumer_thread = threading.Thread(target=func_consumer, args=(q,i,), daemon=True)
        consumer_thread.start()
        consumer_threads.append(consumer_thread)

    producer_thread = threading.Thread(target=func_producer, args=(q,))
    producer_thread.start()

    # 等待完成
    # consumer_thread.join()

    # 等待队列被清空
    q.join()

    print("消费完成")

    pass


from queue import PriorityQueue

def func_PriorityQueue():
    # PriorityQueue 优先队列

    q = queue.PriorityQueue()

    q.put((3, "低优先级"))
    q.put((1, "高优先级"))
    q.put((2, "中优先级"))

    print(q.get())
    print(q.get())
    print(q.get())
    pass


if __name__ == "__main__":
    # func_queue()
    # func_queue_join()
    # func_producer_consumer()
    # func_producer_mult_consumer()
    func_PriorityQueue()

    pass







