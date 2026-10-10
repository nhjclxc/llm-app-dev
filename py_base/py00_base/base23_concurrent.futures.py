
import concurrent.futures
import dataclasses
import os
import pathlib
import random
import time

import httpx

# python 的 concurrent.futures 库必会的方法
# https://docs.python.org/zh-cn/3/library/concurrent.futures.html


# concurrent.futures线程池和进程池： 把“创建线程/进程、提交任务、等待结果”这套操作统一封装成一个更好用的任务池 API。
"""
concurrent.futures
│
├── ThreadPoolExecutor       # 线程池
│
├── ProcessPoolExecutor      # 进程池
│
├── Future                   # 一个异步任务的结果
│   ├── result()
│   ├── done()
│   ├── running()
│   ├── cancelled()
│   ├── cancel()
│   └── add_done_callback()
│
├── wait()
│
└── as_completed()




⭐⭐⭐⭐⭐
ThreadPoolExecutor
ProcessPoolExecutor
submit()
map()
Future.result()
Future.done()
as_completed()

⭐⭐⭐⭐
wait()
Future.cancel()
Future.add_done_callback()

⭐⭐⭐
FIRST_COMPLETED
FIRST_EXCEPTION
ALL_COMPLETED
"""


# 线程池
from concurrent.futures import ThreadPoolExecutor, Future, as_completed, wait

def tpool_init():
    print("tpool init start")
    print("tpool init done")

def tpool_worker(name):
    print("tpool worker start", name)
    time.sleep(1)
    print("tpool worker done")
    return f"tpool worker {name}"

def tpool_worker_rand(name):
    num = random.randint(1,8)
    print("tpool worker start", name, num)
    time.sleep(num)
    print("tpool worker done")
    return f"tpool worker {name} -> {num}"

def func_tpool():

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="tpool_", initializer=tpool_init) as executor:
        # 通过submit提交任务
        # 通过future获取返回值
        future:Future = executor.submit(tpool_worker, "tpool_worker1")
        print("返回结果：", future.result(timeout=1.5))

        # submit 投递单个任务
        # map 批量投递任务
        futures = executor.map(tpool_worker, ["tpool_worker2", "tpool_worker3", "tpool_worker5"])
        for future in futures:
            print("map: ", future)

    # as_completed 哪个任务先完成就先处理哪个任务的结果
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="tpool_", initializer=tpool_init) as executor:

        workers = ["tpool_worker1", "tpool_worker2", "tpool_worker3", "tpool_worker5", "tpool_worker6"]
        futures2: list[Future] = []
        for w in workers:
            futures2.append(executor.submit(tpool_worker_rand, w))
            # Future 完成后自动调用 callback。 任务完成后的回调函数
            # add_done_callback必须要在submit之后立即提交
            # future.add_done_callback(tpool_worker_rand)

        # for future in as_completed(futures2):
        #     print("随机结果：", future.result())

        # wait 通过超时时间来等待一批结果
        done, not_done = wait(futures2, timeout=3)
        for future in done:
            print("done:", future.result())
            future.done()
            future.running()
            # 尝试取消正在运行的任务，但是任务已经开始执行后，不能取消。
            future.cancel()
            # 判断任务是否被取消成功
            future.cancelled()
            #
            pass
        for future in as_completed(not_done):
            print("not_done：", future.result())

        pass



# 进程池
from concurrent.futures import ProcessPoolExecutor


def ppool_init():
    print("ppool init start")
    print("ppool init done")


def ppool_worker_rand(index):
    num = random.randint(1,3)
    print("ppool worker start", index, num)
    time.sleep(num)
    return index * index


def func_ppool():
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = executor.map(ppool_worker_rand, [1,2,3,4,5])
        for future in futures:
            print("map: ", future)
        pass

    pass

# 综合练习


def extract_path_filename0(url):
    path = url.lstrip("https://")
    path = path.lstrip("http://")
    items = path.split("/")
    filename = items[-1]
    path = path.rstrip(f"/{filename}")
    # print("path: ", path)
    # print("filename: ", filename)
    os.makedirs(path, exist_ok=True)
    return path, filename

import os
from urllib.parse import urlparse
from pathlib import Path

def extract_path_filename(url):
    result = urlparse(url)

    host = result.netloc
    path = result.path.lstrip("/")

    filename = Path(path).name
    directory = Path(host) / Path(path).parent

    directory.mkdir(parents=True, exist_ok=True)

    return str(directory), filename
# print(extract_path_filename("https://mirrors.aliyun.com/docker-toolbox/linux/compose/1.10.0/docker-compose-Windows-x86_64.exe"))

def download(url):
    start = time.time()
    print("download", url)
    resp = httpx.get(url)
    path, filename = extract_path_filename(url)
    filepath = os.path.join(path, filename)
    with open(filepath, "wb") as f:
        f.write(resp.content)
        print(f"{filepath} 下载完成")
        return DownloadItem(url, filepath, time.time() - start)

# print(download("https://mirrors.aliyun.com/docker-toolbox/linux/compose/1.10.0/docker-compose-Windows-x86_64.exe"))

@dataclasses.dataclass
class DownloadItem:
    url: str
    path: str
    cost: float

success_list = []

def add_success_url(download_item: DownloadItem):
    success_list.append(download_item)
    pass

def func_httpx():

    urls = [
        "https://mirrors.aliyun.com/docker-toolbox/linux/compose/1.10.0/docker-compose-Darwin-x86_64",
        "https://mirrors.aliyun.com/docker-toolbox/linux/compose/1.10.0/docker-compose-Linux-x86_64",
        "https://mirrors.aliyun.com/docker-toolbox/linux/compose/1.10.0/docker-compose-Windows-x86_64.exe",
        "https://mirrors.aliyun.com/docker-toolbox/linux/compose/1.10.1/docker-compose-Darwin-x86_64",
        "https://mirrors.aliyun.com/docker-toolbox/linux/compose/1.10.1/docker-compose-Linux-x86_64",
    ]
    start = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures: list[Future] = []
        for url in urls:
            futures.append(executor.submit(download, url))
        for future in as_completed(futures):
            # 如果在子线程的执行过程中抛出异常的话，不会在executor.submit的时候抛出，会在读取future的时候抛出，因此在读取future的时候要捕获异常
            try:
                download_item = future.result()
                print("任务执行结果：", download_item)
                add_success_url(download_item)
            except Exception as e:
                print("任务执行失败：", e)
            pass
        pass
    print("执行完毕：", time.time() - start)

if __name__ == '__main__':
    # func_tpool()
    # func_ppool()

    func_httpx()


    pass










