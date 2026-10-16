

import typing

# https://docs.python.org/zh-cn/3/library/typing.html
# python 的 typing 库必会的方法

# 基础类型 → 容器类型 → Optional/Union → Callable → TypeAlias → 泛型 → TypedDict → Protocol → Literal

# 注明参数返回值是什么类型

def add(a: int, b: int) -> int:
    return a + b

# 这个变量是什么类型？
# 这个函数接收什么类型？
# 这个函数返回什么类型？
# 这个字典应该有哪些字段？
# 这个列表里面应该放什么？

# list[int]
# dict[str, int]
# tuple[str, int]
# set[str]

names: list[str] = ["Tom", "Jack", "Alice"]

users: dict[int, str] = {
    1: "Tom",
    2: "Jack"
}

nums: tuple[int, ...] = (1, 2, 3, 4)
user: tuple[int, str] = (1001, "Tom")
tags: set[str] = {"python", "go", "java"}
tags1: set[str] = {"python", "go", "java", 1, 1.2}
for tag in tags1:
    print(tag)


# Optional[str] 表示这个值可能是str类型的数据，也可能是None即空值

from typing import Optional
def get_user(id: int) -> Optional[str]:
    # 返回 str 或 None
    ...
def get_user2(id: int | str) -> str | None:
    # 输入参数id可以是int，也可以是str
    # 返回 str 或 None
    if id == 1:
        return "Tom"
    return None

from typing import Any
# 这个东西可以是任意类型。

# 6. Callable
# 表示：一个可以被调用的对象，也就是函数/可调用对象。

from typing import Callable

# Callable[[参数列表...], 返回值列表]
def execute(func: Callable[[int], str], value: int) -> str:
    return func(value)

def call_func(num: int) -> str:
    return f"{num} is called!"

print(execute(call_func, 666))


# 10. Literal
# 表示：只能是指定的几个值。
from typing import Literal
def set_method(
    method: Literal["GET", "POST"]
) -> str:
    return f"{method.upper()} is called!"

set_method("GET")
# 应为类型 'Literal["GET", "POST"]'，但实际为 'Literal["DELETE"]'
# set_method("DELETE")


from typing import Final
# 这个变量不应该重新赋值。
MAX_CONNECTIONS: Final = 100


from typing import Protocol, runtime_checkable



from typing import overload
# 用于描述：同一个函数，根据不同参数类型有不同返回类型。

@overload
def get_value(x: int) -> int:
    ...

@overload
def get_value(x: str) -> str:
    ...

def get_value(x):
    return x

