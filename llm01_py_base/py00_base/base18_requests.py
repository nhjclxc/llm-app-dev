import json

import requests

# python 的 requests 库必会的方法


# resp = requests.get("http://localhost:8080/health")
resp = requests.get("http://localhost:8080/items/2", timeout=5)

print(resp.status_code)
print(resp.text)

obj = {"name": "cc", "desc": "ccccc"}
data = json.dumps(obj)
resp = requests.post("http://localhost:8080/items", data=data, headers={"Content-Type": "application/json"})
print(resp.status_code)
print(resp.text)
print(resp.json())
# resp.raise_for_status()

print("----------")

# file_url = "https://mirrors.aliyun.com/golang/go1.11.6.src.tar.gz"
# resp = requests.get(file_url, timeout=5, stream=True)
# print(resp.status_code)
# with open("go1.11.6.src.tar.gz", "wb") as f:
#     print("downloading...")
#     download_size = 0
#     for chunk in resp.iter_content(chunk_size=1024*1024):
#         if chunk:
#             f.write(chunk)
#             # f.flush()
#         download_size+= len(chunk)
#     print("downloading bytes: ", download_size, " MB :", download_size / 1024 / 1024 )
#

# import socket
#
# host = "mirrors.aliyun.com"
#
# for x in socket.getaddrinfo(host, 443):
#     print(x)
#
#
# import requests
# import time
#
# file_url = "https://mirrors.aliyun.com/golang/go1.11.6.src.tar.gz"
#
# start = time.time()
#
# resp = requests.get(
#     file_url,
#     timeout=30,
#     stream=True,
# )
#
# resp.raise_for_status()
# #
# # with open("go1.11.6.src.tar.gz", "wb") as f:
# #     for chunk in resp.iter_content(chunk_size=1024 * 1024):
# #         if chunk:
# #             f.write(chunk)
# #
# # elapsed = time.time() - start
# #
# # print(f"耗时: {elapsed:.2f}s")
# # print(f"大小: {resp.headers.get('Content-Length')} bytes")
#
# # 耗时: 61.66s
# # 大小: 21113406 bytes
#
# total = 0
#
# for chunk in resp.iter_content(chunk_size=1024 * 1024):
#     if chunk:
#         total += len(chunk)
#
# elapsed = time.time() - start
#
# print(f"大小: {total / 1024 / 1024:.2f} MB")
# print(f"耗时: {elapsed:.2f}s")
# print(f"速度: {total / elapsed / 1024 / 1024:.2f} MB/s")




import urllib3
import time

http = urllib3.PoolManager()

url = "https://mirrors.aliyun.com/golang/go1.11.6.src.tar.gz"

start = time.time()

resp = http.request(
    "GET",
    url,
    preload_content=False,
)

total = 0

while True:
    chunk = resp.read(1024 * 1024)

    if not chunk:
        break

    total += len(chunk)

elapsed = time.time() - start

print(f"size: {total / 1024 / 1024:.2f} MB")
print(f"time: {elapsed:.2f}s")
print(f"speed: {total / elapsed / 1024 / 1024:.2f} MB/s")

