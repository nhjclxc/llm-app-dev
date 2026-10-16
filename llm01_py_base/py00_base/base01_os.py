
import os

#  https://docs.python.org/zh-cn/3.11/library/os.html


def env():

    name = os.getenv("NAME")
    print("name: ", name)
    print("type: ", type(name))

    print("os.environ[NAME]: ", os.environ["NAME"])

    # os.environ 读取不存在的key会发生：KeyError
    # os.getenv 读取不存在的key返回None
    # print("os.environ[NAME2]: ", os.environ["NAME2"])
    print("name1: ", os.getenv("NAME1"))

    # 不过在正常的 Python 实现里，os.putenv() 和 os.getenv() 通常应该是同步的。
    # 你这个现象最常见的原因是：os.putenv() 并不会更新 Python 的 os.environ 字典，
    # 而 os.getenv() 实际上是从 os.environ 读取的。

    # 不要使用 putenv，
    os.putenv("NAME1", "namename")
    print("name1 1: ", os.getenv("NAME1"))
    # 而是使用os.environ
    os.environ["NAME1"] = "namename"
    print("name1 2: ", os.getenv("NAME1"))

    pass


def path():
    # os.path

    filepath = "/tmp/wb-ext-debug.log"
    print("os.path.exists(filename): ", os.path.exists(filepath))
    print("os.path.isfile(filename): ", os.path.isfile(filepath))

    print("join: ", os.path.join("/tmp", "wb-ext-debug.log"))

    # 获取目录
    print("dirname: ", os.path.dirname(filepath))
    print("abspath: ", os.path.abspath(filepath))
    # 获取文件名
    print("basename: ", os.path.basename(filepath))
    print("splitext: ", os.path.splitext(filepath))

    # 列出目录下的所有文件
    path = "/tmp"
    filename = "wb-ext-debug.log"
    print("listdir: ", os.listdir(path))
    # listdir 只返回目录下的所有文件名，并不返回完整路径，一般要先判断一下在拼接回去
    if filename in os.listdir(path):
        print("filename in os.listdir(path): ", os.path.join(path, filename))

    # 遍历整个目录
    for root, dirs, files in os.walk("/tmp"):
        print("当前目录:", root)
        print("目录:", dirs)
        print("文件:", files)


    pass



def system():

    # os.system("ls -lh")

    import subprocess
    process = subprocess.run(["ls", "-lh"], capture_output=True, text=True)
    print(process.stdout)

    print("os.name: ", os.name)

    import sys
    print(sys.platform)


    pass


def find_logfile(rootpath, extension = None):
    all_files = []
    print("rootpath: ", rootpath)
    print("extension: ", extension)

    if not os.path.exists(rootpath):
        raise FileNotFoundError("目录不存在: " + rootpath)

    # 找出 path 内所有 extension 类型文件
    for root, dirs, files in os.walk(rootpath):
        for file in files:
            # print("file: ", file)
            # names = os.path.splitext(file)
            # if extension is not None and len(names) == 2 and names[1] == extension:
            if extension is not None and file.endswith(extension):
                all_files.append(os.path.join(root, file))
            elif extension is None:
                all_files.append(os.path.join(root, file))

    print("all_files: ", all_files)

    return all_files


def main():

    # env()
    # path()
    # system()

    # find_logfile("/tmp/logs")
    find_logfile("/tmp/logs", ".log")

    pass


if __name__ == '__main__':
    main()

# NAME="lxc" python3 base01_os.py