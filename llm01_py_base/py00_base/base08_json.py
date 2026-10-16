
import json

# https://docs.python.org/zh-cn/3/library/json.html
# python 的 json 库必会的方法


# loads 和 dumps， 后面有个s就是对字符串string的操作
obj = ['foo', {'bar': ('baz', None, 1.0, 2)}]
obj_json_str = json.dumps(obj)
print(obj_json_str)

obj2 = json.loads(obj_json_str)
print(obj2)
print(obj == obj2)
print(obj.__eq__(obj2))

# load 和 dump 对文件的操作
with open('base08_json.json', 'w+', encoding='utf-8') as f:
    json.dump(obj, f)

with open('base08_json.json', 'r+', encoding='utf-8') as f:
    loads_str = json.loads(f.read())
    print("loads_str: ", loads_str)

