

import json

import requests

# pip install requests
# https://github.com/psf/requests
# https://docs.python-requests.org/projects/cn/zh-cn/latest/user/quickstart.html

# 目标服务器：https://httpbin.org/，这个是全球开放的http测试服务端

def exec_get():

    url = "http://127.0.0.1:8080/items"
    resp = requests.get(url)
    print("status_code: ", resp.status_code)
    print("text: ", resp.text)
    print("content: ", resp.content)
    print("headers: ", resp.headers)
    print("reason: ", resp.reason)
    print("apparent_encoding: ", resp.apparent_encoding)
    print("headers['Content-Type']: ", resp.headers['Content-Type'])


def exec_post():

    url = "http://127.0.0.1:8080/items"
    data = {"name": "python2", "desc": "python language"}
    json_str = json.dumps(data)
    print("json_str: ", json_str)
    # data通常用于发送表单数据 或进行文件上传
    # json通常用于发送json数据
    # resp = requests.post(url, data=json_str, headers={"Content-Type": "application/json"})
    resp = requests.post(url, json=data, headers={"Content-Type": "application/json"})
    print("status_code: ", resp.status_code)
    print("text: ", resp.text)
    print("content: ", resp.content)
    print("headers: ", resp.headers)


    pass


def exec_get_id():

    id = 11
    url = f"http://127.0.0.1:8080/items/{id}"
    resp = requests.get(url)
    print("status_code: ", resp.status_code)
    print("text: ", resp.text)
    print("content: ", resp.content)



    pass


def exec_put_id():

    id = 2
    url = f"http://127.0.0.1:8080/items/{id}"
    data = {"name": "python", "desc": "python language hhhhh"}
    resp = requests.put(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    print("status_code: ", resp.status_code)
    print("text: ", resp.text)
    print("content: ", resp.content)


    pass


def exec_url_param():
    url = "http://127.0.0.1:8080/url_param"
    params = {"name": "python", "desc": "python language hhhhh"}
    resp = requests.get(url, params=params)
    print("status_code: ", resp.status_code)
    print("text: ", resp.text, type(resp.text))
    # 响应json内容
    print("json: ", resp.json(), type(resp.json()))
    print("url", resp.url)


    pass


def exec_json():

    r = requests.get('https://api.github.com/events')
    print("r.json(): ", r.json())



    pass


def save_data():
    # 将服务端返回的数据保存起来

    url = "http://127.0.0.1:8080/items"
    resp = requests.get(url)
    print("status_code: ", resp.status_code)
    print("text: ", resp.text)

    chunk_size = 1024
    with open("data.txt", "wb") as f:
        for data in resp.iter_content(chunk_size=chunk_size):
            f.write(data)


def post_file():

    url = "http://127.0.0.1:8080/savefile"
    # files = {"file": open("data.txt", "rb")}
    # 通过数据模拟文件
    files = {'file': ('report.csv', 'some,data,to,send\nanother,row,to,send\n')}
    resp = requests.post(url, files=files, timeout=10)
    print("status_code: ", resp.status_code)
    print("text: ", resp.text)


    pass


def exec_session():
    # 会话对象
    session = requests.Session()
    # 通过session可以为改session下面的请求设置统一数据
    session.headers.update({"auth": "qazqa"})
    session.cookies.update({"aqquth": "qazqa"})

    url = "http://127.0.0.1:8080/items"
    res = session.get(url)
    print("status_code: ", res.status_code)
    print("text: ", res.text)
    print("headers: ", res.headers)
    print("request.headers: ", res.request.headers)

    session.close()

    with requests.Session() as session:
        pass

    pass


def exec_ssl():
    res = requests.get('https://requestb.in', verify=True)
    print("status_code: ", res.status_code)
    print("text: ", res.text)
    # status_code:  200
    # text:  Endpoint not meant for programmatic use. Returning 200 to reduce possible burden.

    res = requests.get('https://requestb.in', verify=True)
    print("status_code: ", res.status_code)
    print("text: ", res.text)

    pass


def main():

    # exec_get()
    # exec_post()
    # exec_get_id()
    # exec_put_id()
    # exec_url_param()
    # exec_json()
    # save_data()
    # post_file()
    # exec_session()
    exec_ssl()
    pass

if __name__ == '__main__':
    main()

