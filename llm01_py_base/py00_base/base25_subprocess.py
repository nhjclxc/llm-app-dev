
# python 的 subprocess 库必会的方法
# https://docs.python.org/zh-cn/3/library/subprocess.html


import subprocess
# subprocess 是 Python 里执行外部命令、启动其他程序、获取命令输出、处理命令退出状态的核心标准库。

"""
⭐⭐⭐⭐⭐
subprocess.run()
subprocess.Popen()
subprocess.PIPE
subprocess.DEVNULL
CompletedProcess
check=True
capture_output=True
text=True
timeout=

⭐⭐⭐⭐
Popen.communicate()
Popen.poll()
Popen.wait()
Popen.terminate()
Popen.kill()

⭐⭐⭐
check_call()
check_output()


"""

# subprocess 是 Python ↔ 操作系统进程之间的桥梁。

subprocess.run(["ls", "-l"])
subprocess.run(["ffmpeg", "-version"])










