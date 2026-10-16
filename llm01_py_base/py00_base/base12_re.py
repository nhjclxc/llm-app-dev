

import re

# https://docs.python.org/zh-cn/3/library/re.html
# python 的 re 库必会的方法
#

# 如果按照“必会”来学，重点掌握：
# re.search()、re.match()、re.fullmatch()、re.findall()、re.finditer()、re.sub()、re.split()、re.compile()
# group()、groups()、groupdict()、span()

# search，在整个字符串中搜索第一个匹配项
text = "https://docs.python.org/zh-cn/321/library/re.html"
# \d表示匹配数字，+表示匹配一个或多个
res = re.search(r"\d+", text)
print(res)
print(res.group())

# match，从字符串头部开始匹配，match只看开头
result = re.match(r"\d+", "789 hello")
print(result)
print(result.group())
# 模式必须和字符串的开头进行匹配
result = re.match(r"\d+", "hello 678")
print(result)
# print(result.group())

# findall，找到所有匹配项，返回 匹配到的list数据。
text2 = "o12ollzzo56oaso89o"
res2 = re.findall(r"\d+", text2)
print(res2)

# finditer，找到所有匹配项，返回 匹配到的 Match 对象迭代器。
for g in re.finditer(r"\d+", text2):
    print(g)


text = "user=zhangsan"
# ()里面的表示捕获组
match = re.search(r"user=(\w+)", text)
print(match.group())
print(match.group(0))
print(match.group(1))

text="user=zhangsan, age=18, addr=beijing"
m = re.match(r"user=(\w+), age=(\d+), addr=(\w+)", text)
print(m)
print(m.group())
print(m.group(1))
print(m.group(2))
print(m.group(3))
print(m.groups())


