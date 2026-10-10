

import csv


# https://docs.python.org/zh-cn/3/library/csv.html
# python 的 csv 库必会的方法
# 最核心的其实就这几个：csv.reader()、csv.writer()、csv.DictReader()、csv.DictWriter()
# 其中我最推荐你重点掌握 DictReader + DictWriter。

# reader 读取到的是列表，DictReader读取到的是字典

data = [
    ["name", "age", "city"],
    ["张三", 20, "北京"],
    ["李四", 25, "上海"],
]

def write_csv():
    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # writer.writerows(data)

        for row in data:
            writer.writerow(row)

# write_csv()

def read_csv():
    with open('data.csv', 'r', newline='') as csvfile:
        # quotechar 表示字符串引号
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            print(row)
# read_csv()

# 字典方式

fieldnames = ["name", "age", "city"]
dict_data = [
    {
        "name": "张三",
        "age": 20,
        "city": "北京"
    },
    {
        "name": "李四",
        "age": 25,
        "city": "上海"
    }
]

def dict_writer():
    with open('dict_data.csv', 'w', newline='') as csvfile:
        w = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 写入头部
        w.writeheader()
        # 写入数据
        w.writerows(dict_data)
        # for row in dict_data:
        #     w.writerow(row)

# dict_writer()

def dict_reader():
    with open('dict_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 默认读取的数据是 <class 'str'> 类型，因此数字类型的数据必须转一次
            age_int = int(row['age'])
            print(row, row['age'], type(row['age']), age_int, type(age_int))
dict_reader()


