
import multiprocessing
import time

# multiprocessing 多进程
# python 的 multiprocessing 库必会的方法
# https://docs.python.org/zh-cn/3/library/multiprocessing.html

# threading：多个线程，共享进程内存
# multiprocessing：多个进程，默认拥有独立内存空间

def worker(aaa):
    print("worker running start: ", aaa)
    # 12. multiprocessing.current_process() 获取当前进程信息
    current_process = multiprocessing.current_process()
    print("current process: ", current_process.pid)
    time.sleep(2)
    print("worker running exit: ")



def func_process():
    # 使用 multiprocessing.Process 创建子进程时，Python 可能会重新导入你的主模块。因此此时必须使用 if __name__ == '__main__': 来运行
    p = multiprocessing.Process(target=worker, args=("ascasc",), name="worker-process")

    print("p.is_alive(): ", p.is_alive())
    p.start()
    print("p.is_alive(): ", p.is_alive(), ", p.name=", p.name)
    if p.is_alive():
        print("p.pid: ", p.pid)

    # terminate()  强制终止子进程：
    time.sleep(1)
    if p.is_alive():
        print("自进程超时运行，强制关闭")
        p.terminate()
        # p.kill()
        print("强制关闭完成")

    p.join()
    print("p.is_alive(): ", p.is_alive())
    # 8. exitcode 查看子进程退出状态：
    print("p.exitcode: ", p.exitcode)
    if p.exitcode == 0:
        print("成功")
    else:
        print("失败")

    # 关闭自进程的相关资源
    p.close()

    print("cpu core number: ", multiprocessing.cpu_count())

    pass

def worker_pool(index):
    print("worker pool start: ", index)
    return index * index

def func_pool():
    # Pool 进程池
    with multiprocessing.Pool(3) as pool:
        # pool.map 同步提交任务
        # results = pool.map(worker_pool, range(10))
        # pool.map_async 异步提交任务
        # results = pool.map_async(worker_pool, range(10))
        # for result in results:
        #     print("result: ", result)

        for i in range(10):
            # apply 表示执行一个任务
            res = pool.apply(worker_pool, args=(i,))
            print("res: ", res)
            # apply_async 异步执行一个任务


    pass


from multiprocessing import Queue
# Queue 用于在多个进程之间传递数据
# 通过 put()、get(timeout) 来控制对列的等待

def pool2_worker(q):
    print("worker pool2 start: ", q.get())
    return 100

def func_pool2():
    #   Queue 这是多进程之间通信非常重要的工具。

    queue = Queue()

    p = multiprocessing.Process(target=pool2_worker, args=(queue,))
    queue.put(123)

    p.start()

    # 防止局部变量 queue 被销毁
    p.join()

def pool3_consumer(q: Queue, res_queue):
    val = q.get(timeout=3)
    print("pool3_consumer: ", val)
    res_queue.put(val* 3)
def pool3_producer(q):
    val = 123
    q.put(val)
    print("pool3_producer : ", val)

def func_pool3():
    #   Queue 这是多进程之间通信非常重要的工具。

    queue = Queue()
    res_queue = Queue()

    p = multiprocessing.Process(target=pool3_producer, args=(queue,))
    p.start()
    c = multiprocessing.Process(target=pool3_consumer, args=(queue,res_queue,))
    c.start()

    c.join()
    p.join()

    print("pool3_producer finished, res_queue: ", res_queue.get())


# 25. Pipe() 两个进程之间直接通信：
# Queue 适用于多生产者、多消费者场景
# Pipe 适用于两个端点的通信

from multiprocessing import Pipe
from multiprocessing.connection import Connection


def pipe_consumer(p: Connection):
    data = p.recv()
    if isinstance(data, int):
        print("data 是 int:", data)
    else:
        print("data 不是 int:", type(data))
        raise ValueError(f"data数据类型{type(data)} 不是需要的类型")
    p.send(data * data)

def pipe_producer(p: Connection):

    p.send(123)

    print("pipe_producer finished: ", p.recv())

def func_pipe():
    parent_conn: Connection
    child_conn: Connection
    # parent_conn 和 child_conn 都会向同一个管道里面传入数据，是Pipe返回的两个不同的端点
    # 一般情况下第一次主动发送消息的可以认为拿第一个parent_conn，第一次主动接收的则拿第二个即可，
    # 两个的功能其实都是一样的，都可以send和recv，因此可以用来做双向通信
    parent_conn, child_conn = Pipe()

    cp = multiprocessing.Process(target=pipe_consumer, args=(child_conn,))
    cp.start()

    pp = multiprocessing.Process(target=pipe_producer, args=(parent_conn,))
    pp.start()


    pp.join()
    cp.join()
    print("func_pipe finished")

    pass


# 使用pipe实现通信

def read_cmd_input(tip) -> str:
    return input(f"{tip}: ")

def point1(name:str, conn: Connection):
    print(f"{name} 已上线")
    while True:
        input_data = read_cmd_input(f"[{name}]请输入：")
        conn.send(input_data)
        if input_data == "exit":
            return

        recv_data = conn.recv()
        print(f"[{name}]收到: {recv_data}")
        if recv_data == "exit":
            return

def point2(name:str, conn: Connection):
    print(f"{name} 已上线")
    while True:
        recv_data = conn.recv()
        print(f"[{name}]收到: {recv_data}")
        if recv_data == "exit":
            return

        input_data = read_cmd_input(f"[{name}]请输入：")
        conn.send(input_data)
        if input_data == "exit":
            return

def func_chat():
    pconn: Connection
    cconn: Connection
    pconn, cconn = Pipe()
    cp = multiprocessing.Process(target=point2, args=("client", cconn,))
    cp.start()
    pp = multiprocessing.Process(target=point1, args=("server", pconn,))
    pp.start()

    pp.join()
    cp.join()
    print("end")

    pass


if __name__ == '__main__':
    # func_process()
    # func_pool()
    # func_pool2()
    # func_pool3()
    # func_pipe()

    name = "test"
    print(read_cmd_input(f"[{name}]请输入"))

    func_chat()


    pass

"""
真正“必会”的 API 如果按照你前面学习标准库的方式，我建议分三个等级。

multiprocessing.Process

p.start()
p.join()
p.is_alive()

p.pid
p.exitcode
p.terminate()

multiprocessing.current_process()
multiprocessing.cpu_count()

multiprocessing.Pool

pool.map()
pool.map_async()

multiprocessing.Queue

queue.put()
queue.get()

multiprocessing.Lock
multiprocessing.Event

"""


"""最后把 multiprocessing 压缩成一张图
multiprocessing
│
├── Process
│   ├── start()
│   ├── join()
│   ├── is_alive()
│   ├── terminate()
│   ├── pid
│   └── exitcode
│
├── Pool
│   ├── map()
│   ├── map_async()
│   ├── apply()
│   ├── apply_async()
│   └── starmap()
│
├── 进程通信
│   ├── Queue
│   │   ├── put()
│   │   └── get()
│   │
│   └── Pipe
│       ├── send()
│       └── recv()
│
├── 同步
│   ├── Lock
│   ├── Event
│   ├── Semaphore
│   └── Barrier
│
├── 共享数据
│   ├── Value
│   ├── Array
│   ├── Manager
│   └── shared_memory
│
└── 基础
    ├── current_process()
    ├── cpu_count()
    └── set_start_method()
"""

