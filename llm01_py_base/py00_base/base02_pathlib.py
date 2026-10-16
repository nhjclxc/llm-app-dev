

import pathlib
import random
import string


# https://docs.python.org/zh-cn/3.13/library/pathlib.html
# https://pypi.org/project/pathlib/
# 优先使用 pathlib 替代 os.path, pathlib更oop

def path():
    filepath = "/tmp/wb-ext-debug.log"
    print(pathlib.Path(filepath))
    print(pathlib.Path(filepath).absolute())
    print(pathlib.Path(filepath).name)
    print(pathlib.Path(filepath).is_file())
    print("pathlib.Path.cwd(): ", pathlib.Path.cwd())

    print("home: ", pathlib.Path.home())

    # 使用path进行路径拼接
    base = pathlib.Path.cwd()
    print(base)
    print(base.joinpath("hello.txt"))
    print(base / "aa" / "bb" / "cc")
    print(base / "aa" / "bb" / "cc" / "ff.txt")

    z = pathlib.Path.cwd() / "base02_pathlib.py"
    print(z.exists())

    # 创建目录
    dir = pathlib.Path.cwd() / "aa" /"bb"
    dir.mkdir(parents=True, exist_ok=True)


    # 遍历目录
    # iter_dir = pathlib.Path("/tmp/logs")
    # for file in iter_dir.iterdir():
    #     if file.is_file():
    #         print("file: ", file)
    #     elif file.is_dir():
    #         print("dir: ", file)
    iterdir("/tmp/logs")


    pass

def iterdir(path):
    iter_dir = pathlib.Path(path)
    for file in iter_dir.iterdir():
        if file.is_file():
            print("file: ", file)
        elif file.is_dir():
            print("dir: ", file)
            iterdir(file)
            # iterdir(file.absolute())


def find_file(dir, glob):
    # 查找目录 dir 下 所有符合 glob 条件的文件

    all_files = []
    path = pathlib.Path(dir)
    for file in path.iterdir():
        if file.is_dir():
            # absolute()：把路径变成绝对路径，但基本不处理符号链接。
            # resolve()：把路径变成绝对路径，并且会尝试解析符号链接，得到真实路径。
            print("dir file.absolute(): ", file.absolute(), ", file.resolve():", file.resolve())
            find_file(file.absolute(), glob)
        if file.is_file() and file.match(glob):
            print("find file: ", file, "; filename: ", file.name, "; name: ", file.stem, "; extension: ", file.suffix)
            all_files.append(file)
            if file.name == "aa.log":
                print("aa.log content: ", file.read_text())
                s = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
                file.write_text(file.read_text() + s)
                print("aa.log content: ", file.read_text())
                # file.read_bytes()
                # file.write_bytes()

    # for file in pathlib.Path(dir).expanduser().glob(glob):
    #     print("file: ", file)

    # 支持递归查找
    # for file in pathlib.Path(dir).expanduser().rglob(glob):
    #     print("rglob: ", file)

    return all_files

def main():


    # path()

    find_file("/tmp/logs", "*.log")

    # 小文件 path.read_text 或 path.read_bytes
    # 处理大文件
    path = pathlib.Path("access.log")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            print(line)

    # 或取相对路径
    path = pathlib.Path("/data/logs/2026/test.log")
    base = pathlib.Path("/data")
    print(path.relative_to(base))

    pass


if __name__ == '__main__':
    main()