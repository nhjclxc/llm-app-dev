import codecs
# python 的 hashlib 库必会的方法
# https://docs.python.org/zh-cn/3/library/hashlib.html

# hashlib 的核心就是：把数据 → 计算固定长度的摘要值。

import hashlib
import json
import secrets
import string
import random
import time


def func_hash1():

    text = b"text"
    print(hashlib.md5(text).hexdigest())
    print(hashlib.sha256(text).hexdigest())
    print(hashlib.sha1(text).hexdigest())


    # 使用 update 往算法里面追加加密数据
    sha = hashlib.sha256()
    sha.update(text)
    print(sha.hexdigest())
    # update 只能接收 bytes 数据
    sha.update(b"aaa")
    sha.update("bbb".encode("utf-8"))
    print(sha.hexdigest())

    # digest() 和 hexdigest() 的区别
    # digest 获取二进制数据，hexdigest 获取十六进制文本数据


    print(hashlib.algorithms_available)

    # 对文件进行hash
    # hashlib.file_digest()

    pass


def func_file_md5():

    md5 = hashlib.md5()
    with open("base27_hashlib.py", "rb") as f:
        while chunk := f.read(1024 * 8):
            md5.update(chunk)
        # for line in f:
        #     md5.update(line)
        pass

    print(md5.hexdigest())
    pass


# python 的 hmac 库必会的方法
def func_hmac():
    # 一般用于接口加密计算
    import hmac

    # 通过一个服务端颁发的key来对客户端的请求信息进行加密
    key = b"secret"
    message = b"hello"

    result = hmac.new(
        key,
        message,
        hashlib.sha256,
    ).hexdigest()
    print("hmac result: ", result)

    result2 = hmac.digest(key, message, hashlib.sha256).hex()
    print("hmac result2: ", result2)

    print("result1 compare result2: ", hmac.compare_digest(result, result2))

    print("hashlib result: ", hashlib.sha256(message).hexdigest())

    pass

import hmac
# HMAC 不是“加密接口信息”，而是对请求内容生成一个“签名/摘要”，用于验证请求是否被篡改，以及请求方是否持有密钥。


_default_size = 10
def gen_random_string(size: int = _default_size):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

def gen_customer_api_key(customer_code: str = None):
    if not customer_code:
        customer_code = gen_random_string()
    customer_code = customer_code.upper()
    return hashlib.md5(customer_code.encode("utf-8")).hexdigest()

# 1ba9834a075f1d902f1fa9cb2effcd69
# abaeec714ed22cd0cb41194d558fa6dd
# print(gen_customer_api_key())

def encoded_body(api_key, timestamp, nonce, body):
    json_body = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    message = f"{timestamp}-{nonce}-{json_body}"

    return hmac.new(api_key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def func_request():

    # 使用 HMAC 对 API 请求进行签名。
    access_key = "1ba9834a075f1d902f1fa9cb2effcd69"
    secret_key = "abaeec714ed22cd0cb41194d558fa6dd"
    timestamp = int(time.time() * 1000)
    nonce = secrets.token_hex(16)
    body = {
        "card_id": "CARD_20260921001",
        "card_type": "借记卡",
        "created_at": "2026-09-21 10:30:00",
    }
    signature = encoded_body(secret_key, timestamp, nonce, body)
    header = {
        "Content-Type": "application/json",
        "Access-Key": access_key,
        "timestamp": timestamp,
        "nonce": nonce,
        "signature": signature,
    }
    print("header: ", header)



    pass


# python 的 base64 库必会的方法
import base64
def func_base64():

    # '+' urlsafe_b64encode 被编码为 '_'
    # '+' b64encode 被编码为 '/'
    url = "https://api.base-64.com/open-api/v1?zhang+san"
    b1 = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8")
    print("b1: ", b1)
    print(base64.urlsafe_b64decode(b1).decode("utf-8"))
    print(base64.b64encode(url.encode("utf-8")).decode("utf-8"))

    # base64.b64encode()  返回的是bytes不是str，因此还要经过一个.decode("utf-8")
    s2 = "string"
    print(base64.b64encode(s2.encode("utf-8")).decode("utf-8"))
    print(base64.b64encode(bytes(s2, "utf-8")).decode("utf-8"))
    print(base64.b64encode(codecs.encode(s2, "utf-8")).decode("utf-8"))

    # base64.b64encode() 对 str 进行编码
    s = b"base64 string"
    s_base64 = base64.b64encode(s).decode("utf-8")
    print(s_base64)
    # base64.b64decode() 对 base64 进行解码
    print(base64.b64decode(bytes(s_base64, "utf-8")).decode("utf-8"))

    # standard_b64encode, standard_b64decode
    print(base64.standard_b64encode(s).decode("utf-8"))

    # 对json数据进行base64编码
    data = {
        "name": "xiaoming",
        "age": 18,
    }
    json_data = json.dumps(data)
    print(json_data)
    base64_json_data = base64.b64encode(json_data.encode("utf-8")).decode("utf-8")
    print(base64_json_data)
    print(base64.b64decode(bytes(base64_json_data, "utf-8")).decode("utf-8"))

    # 对文件进行base64编码
    # /Users/lxc20250729/Downloads/Snipaste_2026-09-21_14-50-36.png
    with open("/Users/lxc20250729/Downloads/Snipaste_2026-09-21_14-50-36.png", "rb") as f:
        base64_png = base64.b64encode(f.read()).decode("utf-8")
        print(base64_png)
        print(f"data:image/png;base64,{base64_png}")
        pass
    pass


if __name__ == "__main__":

    # func_hash1()
    # func_file_md5()
    # func_hmac()
    # func_request()
    func_base64()

    # url = "https://example.com/a/b/c"
    #
    # url_id = hashlib.md5(
    #     url.encode("utf-8")
    # ).hexdigest()
    #
    # print(url_id)
    #
    # # 网页爬虫内容去重复
    # content = "web content"
    #
    # content_hash = hashlib.sha256(
    #     content.encode("utf-8")
    # ).hexdigest()
    #
    #
    pass


"""
hashlib.md5()
hashlib.sha256()
hashlib.sha1()
hashlib.new()

h.update()
h.digest()
h.hexdigest()
"""

