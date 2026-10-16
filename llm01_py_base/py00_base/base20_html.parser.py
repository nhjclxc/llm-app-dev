
# python 的 html.parser 库必会的方法
# https://docs.python.org/zh-cn/3/library/html.parser.html


from html.parser import HTMLParser


class MyParser(HTMLParser):

    def handle_starttag(self, tag, attrs):
        print("开始标签:", tag)
        print("属性:", attrs)

        if tag == "a":
            for name, value in attrs:
                if name == "href":
                    print("tag==a, href:", value)

    def handle_endtag(self, tag):
        print("结束标签:", tag)

    def handle_data(self, data):
        print("文本:", data)

    def handle_startendtag(self, tag, attrs):
        # 处理自结束标签
        print("自结束标签：", tag, attrs)


html = """
<html>
    <body>
        <h1>Hello</h1>
        <a href="https://example.com">Example</a>
    </body>
</html>
"""

parser = MyParser()
parser.feed(html)

print("-"*20)
parser.feed("<p>Hello</p>")
parser.feed('<img src="test.jpg"/>')







