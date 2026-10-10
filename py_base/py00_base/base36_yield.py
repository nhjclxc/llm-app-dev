from pathlib import Path
from typing import Any, Generator


# yield
def gen_num(max_num) -> Generator[int, Any, None]:
    for i in range(max_num):
       #  返回数据之后阻止代码的执行，直至下一次运行到此
       yield i

def test_gen_num() -> None:
    # 返回一个可迭代对象
    gennum = gen_num(10)
    for i in range(5):
        print("next gennum:", next(gennum))
    for num in gennum:
        print("for gennum: ",num)
    pass

def gen_from_num(max_num) -> Generator[int, Any, None]:
    datas = [i for i in range(max_num)]
    yield from datas

def test_gen_from_num() -> None:
    for num in gen_from_num(10):
        print("for gen_from_num: ",num)
    pass

def fib(num) -> Generator[int, Any, None]:
    # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
    a, b = 0, 1
    for i in range(num):
        yield a
        a, b = b, a + b

def test_fib() -> None:
    for num in fib(10):
        print("for fib: ", num)

# yield：一次产出一个值。yield x：把 x 交出去。
# yield from：委托另一个可迭代对象，逐个产出其中的值。yield from x：把 x 里面的元素逐个交出去。
def numbers():
    yield 1
    yield 2

def letters():
    yield "a"
    yield "b"

def combined():
    yield from numbers()
    yield from letters()

# 输出：1 2 a b
# print(list(combined()))

# 练习项目：写一个百万行日志过滤器，要求内存占用基本稳定，并支持中途停止。

from contextlib import closing

def read_log(filepath) -> Generator[str, Any, None]:
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            yield line

def test_read_log() -> None:
    filepath = "base36_yield.py"
    with closing(read_log(filepath)) as lines:
        for line in lines:
            print(line, end="")
            if "from contextlib import closing" in line:
                print("读取到终止标志，提前退出")
                break
    pass



def gen_log(filepath: Path, max_lines: int) -> None:
    # 生成日志
    with filepath.open("w", encoding="utf-8") as f:
        for i in range(max_lines):
            if i % 10 == 0:
                data = f"ERROR 请求失败：{i} \n"
            else:
                data = f"INFO 请求成功：{i} \n"
            f.write(data)
    pass

def test_gen_log(tmp_path: Path) -> None:
    output = Path("base36_yield.log")
    # output = tmp_path / "base36_yield.log"
    gen_log(output, 101)

def _do_match_log(filepath: Path, keyword:str) -> Generator[str, Any, None]:
    # 读取匹配到日志
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            if keyword in line:
                yield line
        pass
    pass
def match_log(filepath: Path, output: Path, keyword:str, max_lines: int = 3) -> list[str]:
    logs = []
    count = 0
    with closing(_do_match_log(filepath, keyword)) as lines:
        with output.open("w", encoding="utf-8") as f:
            for line in lines:
                if keyword in line:
                    logs.append(line)
                    f.write(line)
                    count += 1
                    if count >= max_lines:
                        return logs
    return logs

def test_match_log() -> None:
    input = Path("base36_yield.log")
    output = Path("base36_yield.output.log")
    logs = match_log(input, output, "ERROR", 6)
    print(logs)
    pass

# 处理过滤出来的日志
def deal_log(filepath: Path) -> list[int]:
    indexs: list[int] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            new_line = line.replace("ERROR 请求失败：", "")
            indexs.append(int(new_line))
    return indexs

def test_deal_log() -> None:
    output = Path("base36_yield.output.log")
    indexs = deal_log(output)
    print(indexs)
    pass