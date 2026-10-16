

from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import json


@dataclass
class User:
    id: int
    name: str
    age: int
    email: str


# 模拟数据库
users = {
    123: User(
        id=123,
        name="zhangsan",
        age=25,
        email="zhangsan@example.com",
    ),
    124: User(
        id=124,
        name="lisi",
        age=30,
        email="lisi@example.com",
    ),
}


class UserHandler(BaseHTTPRequestHandler):

    def send_json(self, status_code: int, data: dict):
        body = json.dumps(
            data,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status_code)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        prefix = "/api/v1/user/"

        # 校验请求路径
        if not path.startswith(prefix):
            self.send_json(404, {"error": "not found"})
            return

        id_str = path[len(prefix):]

        # 只接受单个正整数 ID
        if not id_str.isdecimal() or int(id_str) <= 0:
            self.send_json(400, {"error": "invalid user id"})
            return

        user_id = int(id_str)

        # 查询用户
        user = users.get(user_id)
        if user is None:
            self.send_json(404, {"error": "user not found"})
            return

        # 返回用户 JSON
        self.send_json(200, asdict(user))

    def do_POST(self):
        self.send_json(405, {"error": "method not allowed"})

    def do_PUT(self):
        self.send_json(405, {"error": "method not allowed"})

    def do_DELETE(self):
        self.send_json(405, {"error": "method not allowed"})


def main():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 8090),
        UserHandler,
    )

    print("服务启动: http://localhost:8090")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        server.server_close()


# python base37_mypy_pyright_server.py
if __name__ == "__main__":
    main()


