

import argparse

# https://docs.python.org/zh-cn/3/library/argparse.html
# python的 argparse 库必会的方法


# python base06_argparse.py http://baidu.com --host 127.0.0.1 --port 8080 --debug

# 声明一个解析器
parser = argparse.ArgumentParser(
    prog='base06_argparse.py',
    description='What the program does',
    epilog='Text at the bottom of help')


# 定义要解析的参数

# 不带 -- 的参数叫位置参数：
parser.add_argument("url")
parser.add_argument("dest")

# -- 开头的都是要制定参数名称的
# required 指定参数为必填
parser.add_argument("-i", "--ip")
parser.add_argument("--host", type=str, default="8.8.8.8", required=True, help="Host IP address")
parser.add_argument("--port", type=int, default=8090)

# choices：列出所有枚举项
# dest：控制解析之后变量的名字。
parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info", dest="log_level")
# action="store_true"是一个布尔开关, 传入--debug则为True，否则为False。action="store_false"与之相反
parser.add_argument("--debug", action="store_true")
# nargs="+" 可以是的urls变为list接收
parser.add_argument(
    "--urls",
    nargs="+"
)
parser.add_argument(
    "--header",
    action="append"
)

parser.print_help()

# python base06_argparse.py http://baidu.com aliyun.com --host 127.0.0.1 --log-level info --urls "url1" "url2" "url3" --header "h1" --header "h2"

# 执行参数解析
args = parser.parse_args()

# 读取参数
print(args.url)
print(args.dest)
print(args.host)
print(args.port)
print(args.log_level)
print(args.debug)
print(args.urls)
print(args.header)

