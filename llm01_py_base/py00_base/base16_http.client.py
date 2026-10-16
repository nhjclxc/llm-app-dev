
import http.client
import json

# https://docs.python.org/zh-cn/3/library/http.client.html
# python 的 http.client 库必会的方法

# 1. HTTPConnection() —— 创建 HTTP 连接

http_host = "localhost"
http_port = 8080
conn = http.client.HTTPConnection(host=http_host, port=http_port)

# 开启http请求流程
conn.set_debuglevel(1)

# conn.request("GET", "/health")

# obj = {"name": "aa", "desc": "descdesc"}
# json_str = json.dumps(obj)
# conn.request("POST", "/items", json_str, {"Content-Type": "application/json"})


conn.request("GET", "/items/2")

resp = conn.getresponse()
print(resp.status)
print(resp.reason)
# print(resp.read())
# print(resp.read().decode("utf-8"))
print(resp.readline())
print(resp.headers)
print(resp.headers["Content-Type"])
print(resp.getheader("Content-Type"))
print(resp.getheader("content-type"))

conn.close()


# 2. HTTPSConnection() —— 创建 HTTPS 连接

# httpbin.org





