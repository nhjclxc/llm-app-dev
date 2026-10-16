


import urllib.parse
from urllib.parse import urlsplit, urlparse, urlunparse, urlunsplit

# https://docs.python.org/zh-cn/3/library/urllib.parse.html#module-urllib.parse
# python 的 urllib.parse 库必会的方法

# "http://docs.python.org:80/3/library/urllib.parse.html?highlight=params#url-parsing"

url = "http://docs.python.org:80/3/library/urllib.parse.html;ppppp?highlight=params#url-parsing"

# 1. urlparse() —— 解析 URL
up = urlparse(url)
print("up: ", up)

# 2. urlsplit() —— 和 urlparse() 类似
# 将目标url分割为指定的每一个部分
us = urlsplit(url)
print("us: ", us)
print("scheme: ", us.scheme)
print("netloc: ", us.netloc)
print("hostname: ", us.hostname)
print("port: ", us.port)
print("path: ", us.path)
print("query: ", us.query)
print("fragment: ", us.fragment)

# 3. urlunparse() —— 把 URL 组件重新拼成 URL
# 必须传入一个元组，每一个参数的顺序必须对上
# urlunparse((scheme,netloc,path,params,query,fragment))
url2 = urlunparse(("https", "docs.python.org:80", "/3/library/urllib.parse.html", "ppppp", "highlight=params", "url-parsing"))
print("url2: ", url2)

# 4. urlunsplit() —— urlsplit() 的反向操作
# urlunsplit((scheme,netloc,path,query,fragment))
url3 = urlunsplit(("https", "docs.python.org:80", "/3/library/urllib.parse.html;ppppp", "highlight=params", "url-parsing"))
print("url3: ", url3)

# 5. parse_qs() —— 解析 Query 参数，parse_qs() 的 value 永远是 list。
# parse_qs把url中?后面的query参数转化为字典
querys = urllib.parse.parse_qs(url)
print("querys: ", querys)
query = "name=tom&age=20&tag=go&tag=python"
params = urllib.parse.parse_qs(query)
print(params)
print(urllib.parse.parse_qs(us.query))

# 6. parse_qsl() —— Query 参数解析成列表
# parse_qsl 列表里面套元组
# parse_qs 是字典里面套列表
qsls = urllib.parse.parse_qsl(query)
print("qsls: ", qsls)

# 7. urlencode() —— 把参数编码成 Query String
params = {
    "name": "张三",
    "age": 20,
}
encode_params = urllib.parse.urlencode(params)
print("encode_params: ", encode_params)
# 编码出来的参数直接挂在url的path后面，如：url = "https://example.com/search?" + query

# 8. urlencode() 处理重复参数
params2 = [
    ("tag", "go"),
    ("tag", "python"),
    ("tag", "java"),
]
encode_params2 = urllib.parse.urlencode(params2)
print("encode_params2: ", encode_params2)
print("urllib.parse.urlencode(qsls): ", urllib.parse.urlencode(qsls))


# 9. quote() —— URL 编码
# 对url单个参数的编码
eq1 = urllib.parse.quote("参数1")
print("参数1", eq1)
print("参数2", urllib.parse.quote("参数2"))

# 10. unquote() —— URL 解码
print("eq1: ", eq1)
print("unquote(eq1): ", urllib.parse.unquote(eq1))

"""
URL
 │
 ├── urlparse()     → URL 拆开
 │
 ├── urlunparse()   → URL 拼起来
 │
 ├── urljoin()      → URL 路径拼接
 │
 ├── parse_qs()     → Query → dict
 │
 ├── parse_qsl()    → Query → list
 │
 ├── urlencode()    → dict/list → Query
 │
 ├── quote()        → URL 编码
 │
 └── unquote()      → URL 解码

"""

