


# python 的 random 库必会的方法
# https://docs.python.org/zh-cn/3/library/random.html


import random


"""
random()
randint()
randrange()
uniform()
choice()
choices()
sample()
shuffle()
seed()

"""

def func_random():

    # random.random() 生成 [0,1) 之后的随机小数
    r = random.random()
    print(r)
    print(r * 10)
    print(int(r * 10))

    # random.randint 生成 [n,m] 之间的随机整数
    print(random.randint(0, 100))


    # randrange
    print(random.randrange(1, 9, 2))
    print(random.randrange(1, 9, 2))
    print(random.randrange(1, 9, 2))

    # uniform 生成指定范围[a,b]内的随机浮点数。
    print(random.uniform(1.1, 2.2))


    # choice 从指定序列中随机选取一个数据
    datas = ["a", "b", "c"]
    print(random.choice(datas))
    print(random.choices(datas, k = 6))
    # 根据权重随机选择
    print(random.choices(datas, weights=[0.50, 0.30, 0.20], k = 10))

    # sample 随机抽取多个 不重复 的元素，注意choices的抽取可能会重复
    print(random.sample(datas, k = 3))

    # 一个 choice
    # 多个可重复 choices
    # 多个不可重复 sample

    # shuffle 将给定的序列进行打乱，原地修改列表
    print(datas)
    random.shuffle(datas)
    print(datas)

    # 设置随机数种子来获取随机数

    # 以下三个输出每次启动的时候的输出都一样
    # 即：相同 seed -> 相同随机序列
    random.seed(123)
    print(random.randint(1, 100))
    print(random.randint(1, 100))
    print(random.randint(1, 100))


    pass


# python 的 secrets 库必会的方法
# https://docs.python.org/zh-cn/3/library/secrets.html
"""
random
  → 普通伪随机
  → 测试、模拟、抽样、游戏

secrets
  → 密码学安全随机
  → Token、验证码、密码重置链接、Session ID 等
"""
import secrets

def func_secrets():
    items = ["A", "B", "C", "D"]
    print(secrets.choice(items))

    # 2. secrets.randbelow() 生成：0 <= n < upper_bound
    print(secrets.randbelow(10))
    print(secrets.randbelow(10))
    print(secrets.randbelow(10))

    # 生成6位随机验证码
    code = "".join([str(secrets.randbelow(10)) for _ in range(6)])
    print("code: ", code)
    code = secrets.randbelow(1_000_000)
    print(f"{code:06d}")

    token = secrets.token_urlsafe(32)
    print(token)


    pass

if __name__ == '__main__':
    # func_random()
    func_secrets()
    pass

