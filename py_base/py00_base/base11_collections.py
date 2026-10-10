

# collections


# https://docs.python.org/zh-cn/3/library/collections.html
# python 的 collections 库必会的方法
"""
这个模块实现了一些专门化的容器，提供了对 Python 的通用内建容器 dict、list、set 和 tuple 的补充。

namedtuple()一个工厂函数，用来创建元组的子类，子类的字段是有名称的。
deque类似列表的容器，但 append 和 pop 在其两端的速度都很快。
ChainMap类似字典的类，用于创建包含多个映射的单个视图。
Counter用于计数 hashable 对象的字典子类
OrderedDict字典的子类，能记住条目被添加进去的顺序。
defaultdict字典的子类，通过调用用户指定的工厂函数，为键提供默认值。
UserDict封装了字典对象，简化了字典子类化
UserList封装了列表对象，简化了列表子类化
UserString封装了字符串对象，简化了字符串子类化

"""

# Counter → defaultdict → deque → namedtuple → defaultdict 的常见用法 → ChainMap 了解即可

from collections import Counter

# Counter 统计元素出现次数
data = ["a", "b", "a", "c", "a", "b"]
counter = Counter(data)
print(counter)
print(counter.most_common())
counter.update(["b", "b", "c"])
print(counter)
print(counter.most_common(2))


# defaultdict 的核心作用：访问不存在的 key 时，自动创建默认值。
from collections import defaultdict

data = defaultdict(list)
data["name"].append("张三")
data["name"].append("李四")
print(data)

counter = defaultdict(int)
counter["200"] += 1
counter["200"] += 1
counter["404"] += 1
print(counter)

# deque：double-ended queue，双端队列

from collections import deque
deque1 = deque(["a", "b", "c"])
print(deque1)
deque1.appendleft("d")
print(deque1)
deque1.append("f")
print(deque1)
deque1.pop()
print(deque1)
deque1.popleft()
print(deque1)

# 最大存储三个数据，新数据回替换旧数据，将整个数据左移
deque_max = deque(maxlen=3)
for index in range(5):
    deque_max.append(index)
    print(deque_max)