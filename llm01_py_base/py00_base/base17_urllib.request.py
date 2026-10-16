
import urllib.request
from urllib.request import urlopen

# https://docs.python.org/zh-cn/3/library/urllib.request.html
# python 的 urllib.request 库必会的方法





# 1. urlopen() —— 最核心的方法
resp = urlopen("http://localhost:8080/health")
print(resp.status)
print(resp.read().decode("utf-8"))




