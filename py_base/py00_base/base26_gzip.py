


# python 的 gzip 库必会的方法
# https://docs.python.org/zh-cn/3/library/gzip.html

# Python 的 gzip 标准库主要用于读写 .gz 压缩文件。如果你做爬虫、日志处理、CDN 日志分析，这个库非常实用。

"""
| mode   | 含义    |
| ------ | ----- |
| `"rt"` | 文本读取  |
| `"rb"` | 二进制读取 |
| `"wt"` | 文本写入  |
| `"wb"` | 二进制写入 |
| `"at"` | 文本追加  |
| `"ab"` | 二进制追加 |

gzip.open("xxx.gz", "rt", encoding="utf-8")
gzip.open("xxx.gz", "wt", encoding="utf-8")
gzip.open("xxx.gz", "rb")
gzip.open("xxx.gz", "wb")
"""
import gzip


def func_gzip1_write():
    with gzip.open("file.gz", "wt", encoding="utf-8") as f:
        f.write("hello \n")
        f.write("world\n")

        # 把缓冲区的数据刷到文件。
        f.flush()
        pass

    pass

def func_gzip1_read():
    # 1. gzip.open()

    all_lines = []
    with gzip.open('file.gz', 'rb') as f:
        # 一行一行读取， 遍历gz文件
        # 查看读取位置
        print( "tell:", f.tell())
        # print("peek: ", f.peek(5))
        for line in f:
            all_lines.append(line)
            print("line data: ", line)
        print( "tell:", f.tell())
        # line = f.readline()
        # print("line: ", line)
        # readlines() 一次读取所有行
        pass
    print("size: ", len(all_lines))

    # 把数据压缩成gz的二进制数据
    data = b"hello world"
    data_bytes_compressed = gzip.compress(data)
    print( "data_bytes_compressed: ", data_bytes_compressed)
    # 解压被压缩的二进制数据
    de_data = gzip.decompress(data_bytes_compressed)
    print( "de_data: ", de_data)
    print(de_data.decode("utf-8"))


    pass


def func_copy_gzip():

    with gzip.open('src.gz', 'rb') as src:
        with open('dest.gz', 'wb') as dest:
            for line in src:
                dest.write(line)
                pass
            pass
        pass
    pass


if __name__ == "__main__":

    # func_gzip1_write()
    # func_gzip1_read()
    func_copy_gzip()

    pass

