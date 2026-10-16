import json
import time
from wsgiref import headers

import httpx

# python 的 httpx 库必会的方法

"""
| API                           | 用途         |  重要程度 |
| ----------------------------- | ---------- | ----: |
| `httpx.get()`                 | GET 请求     | ⭐⭐⭐⭐⭐ |
| `httpx.post()`                | POST 请求    | ⭐⭐⭐⭐⭐ |
| `httpx.put()`                 | PUT 请求     |  ⭐⭐⭐⭐ |
| `httpx.patch()`               | PATCH 请求   |  ⭐⭐⭐⭐ |
| `httpx.delete()`              | DELETE 请求  |  ⭐⭐⭐⭐ |
| `httpx.request()`             | 通用 HTTP 请求 | ⭐⭐⭐⭐⭐ |
| `httpx.Client()`              | 同步客户端、连接复用 | ⭐⭐⭐⭐⭐ |
| `httpx.AsyncClient()`         | 异步 HTTP 请求 | ⭐⭐⭐⭐⭐ |
| `response.status_code`        | HTTP 状态码   | ⭐⭐⭐⭐⭐ |
| `response.text`               | 响应文本       | ⭐⭐⭐⭐⭐ |
| `response.json()`             | JSON 响应    | ⭐⭐⭐⭐⭐ |
| `response.content`            | 二进制响应      |  ⭐⭐⭐⭐ |
| `response.headers`            | 响应头        | ⭐⭐⭐⭐⭐ |
| `response.cookies`            | Cookie     |   ⭐⭐⭐ |
| `response.url`                | 最终 URL     |   ⭐⭐⭐ |
| `response.raise_for_status()` | HTTP 错误检查  | ⭐⭐⭐⭐⭐ |
| `httpx.Timeout`               | 超时控制       | ⭐⭐⭐⭐⭐ |
| `httpx.Limits`                | 连接池限制      |  ⭐⭐⭐⭐ |


"""

# 2. httpx.get()

# 3. httpx.post()
# data可以直接发送表单数据
# json则发送json数据

timeout = httpx.Timeout(connect=5, read=10, write=10, pool=5)

obj = {"name": "cc", "desc": "ccccc"}
url = "http://localhost:8080/items"
resp = httpx.post(url, json=obj, headers={"Content-Type": "application/json"}, timeout=timeout)
print(resp.status_code)
print(resp.text)
print(resp.json())


"""
timeout = httpx.Timeout(
    connect=5,
    read=10,
    write=10,
    pool=5,
)

| 类型        | 含义            |
| --------- | ------------- |
| `connect` | 建立 TCP/TLS 连接 |
| `read`    | 等待服务器返回数据     |
| `write`   | 向服务器发送数据      |
| `pool`    | 从连接池获取连接      |

"""

print("-"*55)

get_id = resp.json()["id"]
get_url = f"http://localhost:8080/items/{get_id}"
resp = httpx.get(get_url)
print(resp.status_code)
print(resp.text)
print(resp.json())
print(resp.url)
print(resp.history)
print(resp.is_success)
print(resp.is_error)
# 如果：200 正常继续。如果：404、500 会抛异常。
resp.raise_for_status()


# start = time.time()
# stream_url = "https://mirrors.aliyun.com/golang/go1.11.6.src.tar.gz"
# with httpx.stream("GET", stream_url) as response:
#     print(response.status_code)
#     with open("go1.11.6.src.tar.gz", "wb") as f:
#         for chunk in response.iter_bytes():
#             f.write(chunk)
# end = time.time()
# print(end - start)

#
# try:
#     resp = httpx.get("http://www.baidu.com")
#     resp.raise_for_status()
#     print(resp.text)
# except httpx.HTTPStatusError as e:
#     print("请求出错：", e)

# 控制 tcp 连接池
limits = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20,
)
# 使用client进行复用tcp连接
def use_client():
    print("使用 httpx.Client 进行请求")
    headers = {
        "Authorization": "Bearer xxx",
        "User-Agent": "my-client/1.0",
    }
    with httpx.Client(timeout=timeout, headers = headers, limits=limits) as client:
        try:
            # 新增
            obj = {"name": "client cc", "desc": "client ccccc"}
            post_url = "http://localhost:8080/items"
            resp = client.post(post_url, json=obj)
            resp.raise_for_status()
            print(resp.status_code)
            print(resp.json())

            # 获取
            get_id = resp.json()["id"]
            get_url = f"http://localhost:8080/items/{get_id}"
            resp = client.get(get_url)
            resp.raise_for_status()
            print(resp.status_code)
            print(resp.json())

        except httpx.HTTPStatusError as ee:
            print(f"请求出错：{ee}")
    print("-----")

use_client()

import asyncio


async def httpx_async(get_id: int):
    # 调用 httpx.AsyncClient() 的外部方法必须是异步方法

    # httpx.AsyncClient() 是 httpx 相对于传统 requests 非常重要的能力。
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        try:
            get_url = f"http://localhost:8080/items/{get_id}"
            # 异步等待
            print("异步接口开始 ", get_id)
            resp = await client.get(get_url)
            print("异步接口结束返回", get_id)
            resp.raise_for_status()
            print(resp.status_code)
            print(resp.json())
        except httpx.HTTPStatusError as ee:
            print("调用出错：", ee)


async def do_async():
    print("---开始异步并发调用---")
    # 这三个 httpx_async() 都是 coroutine。
    # gather() 会把它们纳入 asyncio 的调度，并等待它们全部完成。
    await asyncio.gather(
        httpx_async(1),
        httpx_async(2),
        httpx_async(3),
    )
    print("---异步并发调用结束---")

print("start")

asyncio.run(do_async())

print("done")


# 使用http2需要安装：pip install "httpx[http2]"

# httpx.RequestError
# httpx.HTTPStatusError
# httpx.TimeoutException



# resp = httpx.get("http://61.163.231.164:18090/static/tavern/")
# print(resp.status_code)
# print(resp.text)

print("302跟随开始")

url302 = "http://localhost:8080/jump302"
# follow_redirects=False不回执行location里面返回的连接，则True的时候会直接执行里面的连接
resp = httpx.get(url302, follow_redirects=True)
print("状态码：",resp.status_code)
print("返回内容", resp.text)
print("返回location：", resp.headers.get("location"))


print("302跟随结束")



class GoAPIClient:
    _get_url = "/items"
    _save_url = "/items"
    def __init__(self, base_url):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "my-client/1.0",
        }
        self.timeout = httpx.Timeout(connect=5, read=10, write=10, pool=5)
        self.limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        self.client = httpx.Client(timeout=self.timeout, limits=self.limits, headers=self.headers)

    def close(self):
        self.client.close()
        print("client 资源关闭")

    def get(self, id):
        if self.client.is_closed:
            print("客户端连接已关闭")
            return None
        req_url = f"{self.base_url}{GoAPIClient._get_url}/{id}"
        print("req_url: ", req_url)
        resp = self.client.get(req_url)
        resp.raise_for_status()
        return resp.json()

    def save(self, obj):
        if self.client.is_closed:
            print("客户端连接已关闭")
            return None
        req_url = f"{self.base_url}{GoAPIClient._save_url}"
        resp = self.client.post(req_url, json=obj)
        resp.raise_for_status()
        return resp.json()

client = GoAPIClient(base_url="http://localhost:8080")
save_resp = client.save({"name": "GoAPIClient cc", "desc": "GoAPIClient ccccc"})
print(save_resp)
print("client.get(1): ", client.get(save_resp["id"]))

client.close()
print("client.get(1): ", client.get(save_resp["id"]))



