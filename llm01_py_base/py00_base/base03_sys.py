
import sys

# sys --- 系统相关的形参和函数
# https://docs.python.org/zh-cn/3/library/sys.html
# os 主要处理操作系统；pathlib 主要处理文件路径；sys 主要处理 Python 解释器本身、程序运行环境以及命令行参数。
# python的sys库必会的方法

# 读取命令行参数
# python base03_sys.py hello 123
print(sys.argv)
# ['base03_sys.py', 'hello', '123']
print(sys.argv[0])

#
# sys.exit(-1)

# py模块搜索路径
# 当 执行 import xxx 的时候py解释器根据这个路径去搜索相关模块
print(sys.path)

# sys.path.append() —— 添加模块搜索路径

# 当前py解释器路
print(sys.executable)

import os
os.system("which python")

import subprocess
process = subprocess.run(["which", "python"], capture_output=True, text=True)
print(process.stdout)

# 当前操作系统
print(sys.platform)
if sys.platform == "darwin":
    print("Mac")

elif sys.platform.startswith("linux"):
    print("Linux")

elif sys.platform == "win32":
    print("Windows")






# sys.stdin.readline()
for line in sys.stdin:
    print(line, end="")
    if line == "\n":
        break














