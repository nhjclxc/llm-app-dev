import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Protocol
from unittest.mock import Mock

import httpx
import pytest


# python -m pip install mypy pyright
# uv add mypy pyright



def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> str:
    return f"Hello, {name}"

def test_1() -> None:
    print(add(10, 20))
    print(greet("Python"))

    # 故意制造错误
    # print(add("10", 20))
    # print(greet(123))


# uv run mypy base37_mypy_pyright.py
# uv run pyright base37_mypy_pyright.py

from dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str
    age: int
    email: Optional[str] = None

def describe_user(user: User) -> str:
    return f"{user.name}, age={user.age}"

class UserRepositoryInstance:
    def __init__(self, users: List[User]):
        self.user_map = {}
        for user in users:
            self.user_map[user.id] = user

    def get_user_by_id(self, id: int) -> Optional[User]:
        return self.user_map.get(id)

@pytest.fixture(scope="function")
def users() -> List[User]:
    user1 = User(1, "Alice", 25)
    user2 = User(2, "tom", 35, "tom@gmail.com")
    return [user1, user2]

def test_user(users) -> None:
    print(describe_user(users[0]))
    print(describe_user(users[1]))
    user_repository = UserRepositoryInstance(users)
    print(user_repository.get_user_by_id(2))
    print(user_repository.get_user_by_id(3))


class UserRepository(Protocol):
    def get_user_by_id(self, id: int) -> User | dict[str, object] | None:
        """
        通过 ID 查找用户。

        Args:
            id: 用户 ID。

        Returns:
            用户实体、字典或 None（用户不存在时）。
        """
        # pass 或 ... 表示没有实际的实现体，由具体的实现类去实现
        # pass
        ...

class InMemoryUserRepository(UserRepository):
    def __init__(self, users: List[User]):
        self.user_map = {}
        for user in users:
            self.user_map[user.id] = user

    def get_user_by_id(self, id: int) -> Optional[User | dict[str, object] | None]:
        return self.user_map.get(id)

def test_in(users)-> None:
    user_repository = InMemoryUserRepository(users)
    print(user_repository.get_user_by_id(2))
    print(user_repository.get_user_by_id(3))



class SqliteUserRepository(UserRepository):
    def __init__(self, filepath: Path):
        self.conn = sqlite3.connect(filepath)
        self.__init_table()

    def __init_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
              CREATE TABLE IF NOT EXISTS USER
              (
                 ID      INT PRIMARY KEY NOT NULL,
                 NAME    TEXT            NOT NULL,
                 AGE     INT             NOT NULL,
                 EMAIL   CHAR(50)
              );''')
        self.conn.commit()
        print("数据库创建成功")

    def close(self):
        self.conn.close()
        self.conn = None

    def add(self, id0: int, name: str, age: int, email: Optional[str| None]) -> None:
        if self.conn is None:
            raise Exception("数据库连接已关闭")

        cursor = self.conn.cursor()
        id = id0 + int(time.time())
        cursor.execute("insert into USER (ID, NAME, AGE, EMAIL) values (?, ?, ?, ?)", (id, name, age, email))
        self.conn.commit()


        pass

    def get_user_by_id(self, id: int) -> Optional[User | dict | None]:
        cursor = self.conn.execute("select ID, NAME, AGE, EMAIL from USER where ID = ?", (id,))
        # x= cursor.fetchone()
        for row in cursor:
            return User(row[0], row[1], row[2], row[3])

def test_SqliteUserRepository(users):

    db_path = Path("test.db")
    repo = SqliteUserRepository(db_path)
    for user in users:
        repo.add(user.id, user.name, user.age, user.email)

    print(repo.get_user_by_id(1))
    print(repo.get_user_by_id(2))
    print(repo.get_user_by_id(3))

    repo.close()

# 让业务函数依赖接口，而不是具体实现
def get_user(user_repo: UserRepository, user_id: int) -> Optional[User | dict[str, object] | None]:
    return user_repo.get_user_by_id(user_id)

def test_get_user(users):

    memory_repo = InMemoryUserRepository(users)
    print(get_user(memory_repo, 1))

    sqlite_repo = SqliteUserRepository(Path("test.db"))
    print(get_user(sqlite_repo, 1))

    pass

def get_user1(id: int) -> tuple[int, str]:
    """
    返回用户
    Args:
        id: 用户id
    Returns:
        tuple[int, str]: 包含用户 ID 和用户名字的元组。
    """
    return 1, "aa"

def get_user12(id: int):
    return 1, "aa"

# user_id, name = get_user1(1)
#
# print(user_id)  # 1
# print(name)     # aa



# 综合项目——类型安全的用户服务
#
# 项目要求
# 1、使用 dataclass 或 Pydantic 定义 User 模型
# 2、使用 httpx 实现 HTTP 客户端
# 3、使用 Protocol 定义数据访问接口
# 4、使用 Optional 或联合类型处理用户不存在的情况
# 5、使用 async/await 实现异步查询
# 6、使用 pytest 和 Mock 编写单元测试
# 使用 mypy 与 pyright 检查整个项目
# 在 CI 中执行测试和类型检查

@dataclass
class User:
    id: int
    name: str
    age: int
    email: Optional[str] = None

class Client:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=httpx.Timeout(10),
        )
    def close(self):
         self.client.close()

    def request(self, id: int) -> dict | None:
        req_url = f"{self.base_url}/api/v1/user/{id}"
        resp = self.client.get(req_url, timeout=5)
        resp.raise_for_status()
        return resp.json()

def test_client() -> None:
    client = Client("http://127.0.0.1:8090")
    res = client.request(123)
    user = User(res.get("id"), res.get("name"), res.get("age"), res.get("email"))
    print(user)
    pass

class RequestInterface(Protocol):
    def get_by_id(self, client: Client, id: int) -> User | None:
        ...

class AsyncRequest(RequestInterface):
    def get_by_id(self, client: Client, id: int) -> User | None:
        user = self._get_by_id(client, id)
        return user

    async def _get_by_id(self, client: Client, id: int) -> User | None:
        async_tasks = asyncio.create_task(client.get_by_id(id))
        res = await asyncio.gather(async_tasks)
        return res


def test_cc() -> None:

    mock_client = Mock()
    mock_client.get.response = httpx.Response(status_code=200, json="{'id': 1, 'name': 'Alice', 'age': 25, 'email': 'alice@gmail.com'}")

    arequest = AsyncRequest()
    res = arequest.request(mock_client, 1)
    print(res)

    pass

