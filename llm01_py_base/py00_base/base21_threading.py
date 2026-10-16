
# python 的 threading 库必会的方法
# https://docs.python.org/zh-cn/3/library/threading.html


import threading
import time


def work1(name):
    time.sleep(1)
    print(f"work {name}", threading.current_thread())

def t1():
    thread = threading.Thread(
        target=work1,
        name="download-worker",
        args=("zhangsan",)
    )
    print("# 开启线程")
    thread.start()
    print("# 等待线程结束")
    print("thread.is_alive() ", thread.is_alive())
    thread.join(timeout=3)
    print("thread.is_alive() ", thread.is_alive())
    print("执行完毕")

    main_thread = threading.main_thread()

    print(main_thread.name)

# t1()

# lock的使用

counter = 0
# work2 里面由于 counter 是多线程共享变量，很可能存在同时写的问题，
def work2(name):
    global counter
    for i in range(10):
        time.sleep(0.1)
        counter += 1
    print(f"work {name}: counter: {counter}")

lock = threading.Lock()
def work_lock(name):
    global counter
    # with lock 包含 lock.acquire() 和 lock.release() 两个过程
    with lock:
        for i in range(10):
            time.sleep(0.1)
            counter += 1
        print(f"work {name}: counter: {counter}")

# RLock 可重入锁，可以在线程内部被多次获取锁
rlock = threading.RLock()
def func1():
    with rlock:
        func2()
def func2():
    with rlock:
        print("hello")

def t2():
    threads = []

    for i in range(3):
        t = threading.Thread(
            target=work_lock,
            args=(i,),
            name=f"download-lock-{i}",
        )
        t.start()# 先全部启动
        threads.append(t)

    for t in threads:
        t.join()# 再统一等待

    pass
# t2()


# 3、Event, 信号量
event = threading.Event()
# 线程等待
# event.wait()
# 通知线程运行, 所有正在 event.wait() 的线程都会被唤醒执行
# event.set()
# 清除事件，之后所有 event.wait() 的又会进入等待，其实clear就是set的反面，set是让所有线程都通过wait，而clear则是让所有线程都卡在wait
# 应用场景，某个外部资源还没准备好的时候使用clear让所有线程等待，当外部资源准备好了之后使用set通知所有线程往下执行。（当再次等待外部资源时还是使用clear来进行等待）
# event.clear()

def func3_work():
    print("进入 func3_work")
    print("等待信号量")
    event.wait()
    print("func3_work done")

def func3():
    t = threading.Thread(target=func3_work)
    t.start()
    time.sleep(2)
    print("main send event")
    event.set()
    t.join()

# func3()

class DataObj:
    def __init__(self):
        self.data = ""
        self._count = 0
        self._file_count = 0
        self._lock = threading.Lock()
    def set_data(self, data):
        self.data = data
    def counter(self):
        global done
        with self._lock:
            self._count += 1
            if self._count % 3 == 0:
                self._file_count += 1
                data_event.clear()
                read_data_event.set()

data = DataObj()


data_event = threading.Event()
read_data_event = threading.Event()
# data_event.clear() # event默认时False，一创建的时候可以不用使用clear


files = ["f1", "f2", "f3"]
def io_work():
    global data
    while data._file_count < len(files):
        read_data_event.clear()
        print(f"开始数据读取")
        time.sleep(1.25)
        with open(files[data._file_count], "r", encoding="utf-8") as f:
            data.set_data(f.read())
            print(f"{files[data._file_count]} 数据读取完毕，设置event")
            data_event.set()
            read_data_event.wait()
    else:
        print("读取数据结束")
# io_work()
# print(data)

def func5_work(index):
    while data._file_count < len(files):
        print(f"func5_work_{index}, 等待中...")
        data_event.wait()
        print(f"func5_work_{index}, data: {data.data}")
        data.counter()
    else:
        print(f"func5_work_{index} done")

def func5():
    threads = []

    # 先创建所有消费线程
    for i in range(3):
        t = threading.Thread(target=func5_work, args=(i,))
        t.start()
        threads.append(t)

    # 开启生产者线程
    producer = threading.Thread(target=io_work)
    producer.start()
    producer.join()

    # 等待消费线程执行结束
    for t in threads:
        t.join()
    pass
# func5()

# Condition 用于线程之间等待某个条件发生。
# Event 是“有没有发生这件事”的信号；Condition 是“某个条件是否满足”的等待/通知机制。
"""
|          | Event          | Condition                     |
| -------- | -------------- | ----------------------------- |
| 核心       | 信号/状态          | 条件                            |
| 状态       | set / clear    | 通常由共享数据决定                     |
| 是否需要锁    | 内部自己处理         | 通常需要配合 Lock                   |
| wait     | 等待 Event 被 set | 等待 notify/条件满足                |
| notify   | 没有             | 有 `notify()` / `notify_all()` |
| 典型场景     | “准备好了”         | “队列有数据了”                      |
| 是否适合多个条件 | 不太适合           | 非常适合                          |

"""

def func_event():

    event = threading.Event()

    datas = []

    def consumer():
        event.wait()
        d = datas.pop()
        print(f"event 数据：{d}")

    def producer():
        datas.append("666")
        event.set()

    t = threading.Thread(target=consumer)
    t.start()

    producer()

    t.join()

# func_event()

def func_cond():

    cond = threading.Condition()

    datas = []

    def consumer():
        with cond:
            while not datas:
                cond.wait()

        d = datas.pop()
        print(f"cond 数据：{d}")

    def producer():
        with cond:
            datas.append("666")
            cond.notify()

    t = threading.Thread(target=consumer)
    t.start()

    producer()

    t.join()

# func_cond()


# 22. Semaphore: 限制同时运行的线程数量。

sem = threading.Semaphore(3)


def func8_work(index):
    # with sem:
    #     print(f"func8_work: {index} start")
    #     time.sleep(1)
    #     print(f"func8_work: {index} end")
    sem.acquire()
    print(f"func8_work: {index} start")
    time.sleep(1)
    print(f"func8_work: {index} end")

    sem.release()

def func8():
    threads = []
    for i in range(5):
        t = threading.Thread(target=func8_work, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # func8_work: 0 start
    # func8_work: 1 start
    # func8_work: 2 start
    # func8_work: 0 end
    # func8_work: 1 end
    # func8_work: 2 end
    # func8_work: 3 start
    # func8_work: 4 start
    # func8_work: 3 end
    # func8_work: 4 end
    pass
# func8()

# 24. Timer 延迟执行：
# exit_cond = threading.Condition()
def func9_work():
    print("func9_work")
    # with exit_cond:
    #     exit_cond.notify()

def func9():
    timer = threading.Timer(1, func9_work)
    timer.start()
    # 等待执行完毕
    timer.join()
    # with exit_cond:
    #     exit_cond.wait()
    #     print("exit")

    # timer.cancel() 可以取消还未执行的timer
    pass
# func9()

# 25. daemon  这个虽然不是方法，但非常重要。
# Daemon 线程：主线程结束后，Python 不会因为它还在运行而继续等待整个程序退出。
# daemon=True 表示后台运行的线程, 在没有t.join的时候主线程不回等待这个自线程的执行完毕
# daemon 线程不会被优雅地等待完成。
# 所以需要保存数据、关闭资源的线程，不应该简单设置成 daemon。

def func10():
    def func10_daemon():
        print("func10_daemon enter")
        time.sleep(3)
        print("func10_daemon exit")
    t = threading.Thread(target=func10_daemon, daemon=True)
    t.start()
    t.join()

    print("func10 exit")

    pass

# func10()

# 26. threading.local() 线程本地存储。
# 线程局部存储（Thread Local Storage，TLS）
# 核心作用： 同一个 local 对象，不同线程看到的是不同的数据。

local = threading.local()
def func11_work(index):
    local.name = f"func11_work-{index}"
    print(threading.current_thread().name, local.name)

def func11():

    threads = []
    for i in range(5):
        t = threading.Thread(target=func11_work, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
# func11()


# 创建线程池 from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

urls = [f"http://examole.com/{i}" for i in range(10) ]
def func12_work(url):

    print(f"{threading.current_thread().name}: {url} enter")
    time.sleep(1)
    print(f"{threading.current_thread().name}: {url} leave")
    return f"{threading.current_thread().name}: {url} exec"

def func12():
    with ThreadPoolExecutor(max_workers=3) as executor:
        # executor.map会依次把 urls 里面的每一个数据传入 func12_work
        # 类似于： executor.submit(func12_work, urls[0])， executor.submit(func12_work, urls[1]) ...
        # results = executor.map(func12_work, urls)
        # for result in results:
        #     print(result)

        futures = []
        for url in urls:
            future = executor.submit(func12_work, url)
            futures.append(future)
        for future in futures:
            print(future.result())
        pass

func12()



