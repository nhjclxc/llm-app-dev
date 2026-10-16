import json
import sqlite3
import time

import httpx
import pytest


def add(a: int, b: int) -> int:
    return a + b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


def test_add():
    # assert 后面接布尔表达式

    assert add(1, 2) == 3

def test_divide():
    assert divide(1, 2) == 0.5

def test_divide0():
    # pytest.raises 捕获异常，
    #   ValueError 表示异常类型
    #   match="除数不能为 0" 表示要匹配的异常消息
    with pytest.raises(ValueError, match="除数不能为 0"):
        divide(1, 0)

def test_divide02():
    # pytest.raises 捕获异常，
    #   ValueError 表示异常类型
    #   match="除数不能为 0" 表示要匹配的异常消息
    with pytest.raises(ValueError) as err:
        divide(1, 0)
    assert str(err.value.args[0]) == "除数不能为 0"

class TestUser:
    def test_name(self):
        assert "tom".upper() == "TOM"


def test_collection() -> None:
    results = ['result', 'success']
    assert results
    assert len(results) == 2
    assert results[0] == 'result'
    assert 'success' in results

    age = 18
    assert age == 18, f"实际年龄错误：{age}"

    pass


# @pytest.fixture
# 用于准备测试依赖数据
# function, 默认每一个函数执行签都调用
# class，每个测试类一次
# module，每个模块一次
# session，整个测试会话一次
@pytest.fixture(scope="function")
def gen_user() -> dict[str, object]:
    print("用于准备测试依赖数据 @pytest.fixture 111")
    return {"id": 1, "name": "Tom", "age": 18}


# 常用内置 fixture：
# - `tmp_path`：当前测试独立的临时目录；
# - `tmp_path_factory`：创建会话级临时目录；
# - `monkeypatch`：临时修改环境变量、属性和字典；
# - `capsys`：捕获标准输出和标准错误；
# - `caplog`：捕获日志；
# - `request`：访问当前测试的元信息；
# - `pytestconfig`：访问 pytest 配置。
@pytest.fixture(scope="function")
def fixture0(tmp_path, tmp_path_factory) -> None:

    print("临时目录 tmp_path:", tmp_path)
    print(" tmp_path_factory:", tmp_path_factory)
    pass

# 被 @pytest.fixture 装饰器修饰的函数不能直接运行，必须通过某个 test_方法来注入运行
def test_fixture0(fixture0,tmp_path):
    print(1)
    print("tmp_path --> ", tmp_path)

    pass

# 使用 monkeypatch 修改临时变量
def get_url() -> str:
    import os
    return os.environ.get("API_PATH")
    # return "http://localhost:8080"

def test_monkeypatch(monkeypatch) -> None:
    # 修改 API_PATH 环境变量
    excepted = "http://localhost:8080/123"
    monkeypatch.setenv("API_PATH", excepted)
    result = get_url()
    print(result)
    assert result == excepted


def test_pytestconfig(pytestconfig) -> None:
    print(pytestconfig)
    pass

# 使用 @pytest.fixture 进行测试数据管理
@pytest.fixture(scope="function")
def create_file(tmp_path) -> None:

    print("tmp_path:", tmp_path)
    # tmp_path = tmp_path / "test.txt"
    # with open(tmp_path, "w") as f:
    #     f.write("123qwe")
    pass

def test_collection2(gen_user) -> None:
    # user 将 gen_user 生成的数据返回
    print("u.id = ", gen_user.get("id"))
    age = 18
    assert age == 18, f"实际年龄错误：{age}"
    pass


@pytest.fixture(scope="function")
def conn():
    """ conn 名称必须和使用时候的名称一致 """
    conn = sqlite3.connect('test.db')
    try:
        yield conn
    finally:
        conn.close()
        print("连接关闭")

# 直接通过 @pytest.fixture 来注入 conn，这个conn和装饰器下面定义的名称必须一致
def test_conn(conn) -> None:
    cursor = conn.cursor()
    cur = cursor.execute('select * from COMPANY')
    for row in cur:
        print(row)
    pass



def len_count(text:str) -> int:
    return len(text)

## 8. 参数化测试
# 当同一逻辑需要测试多组输入时，使用 `@pytest.mark.parametrize`：
# 使用多组数据，重复执行同一个测试函数。
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("az", 2),
        ("abc", 3),
    ],
)
def test_text_length(text: str, expected: int, user_data) -> None:
    print("user_data . ", user_data)
    assert len_count(text) == expected

def judge(text) -> bool:
    return text == "pass"

# 使用 `pytest.param` 添加 ID 或标记：
@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("pass", True, id="empty string"),
        pytest.param("unpass", False, id="empty string2"),
    ]
)
def test_text_length2(text: str, expected: bool) -> None:
    assert judge(text) == expected

# conftest.py 里面的 fixture 可以被 conftest.py 所在的文件夹内部所有的 test_测试用例发现并可以使用
def test_user_data(user_data: dict[str, object]) -> None:
    print("user_data . ", user_data)
    pass


# ## 12. unittest.mock：模拟依赖
# 当代码依赖 HTTP、邮件、支付、时间或第三方 SDK 时，不应让单元测试真的调用这些服务。
# 使用 from unittest.mock import Mock 来代替
from unittest.mock import Mock


def send_notice(mailer: Mock, dest_email, content):
    """
        发送通知方法
        mailer: 表示发送对象
        dest_email: 目标地址
        context: 内容
    """

    print(f"发送对象：{mailer}, 发送目标地址：{dest_email}, 发送内容：{content}。")

    # 五、规则 4：Mock 记录的是调用行为，不会自动执行真实业务
    mailer.send(dest=dest_email, content=content)

    pass

# 模拟http等调用
def test_send_notice() -> None:

    # mock对象，并非真的发送对象
    mailer = Mock()

    # 调用业务函数, 这个就是要测试的函数
    send_notice(mailer, "tom@example.com", "hello")

    # 记录方法调用，即验证方法行为
    mailer.send.assert_called_once_with(
        dest="tom@example.com",
        content="hello",
    )

    pass


def load_user(client, id) -> dict[str, object]:
    result = client.get(id=id)
    return result

def load_user2(client, id) -> dict[str, object]:
    result = client.geqt(id=id)
    return result

# 方法        作用
# assert_called() 验证至少调用过一次
# assert_not_called() 验证从未调用
# assert_called_once() 验证恰好调用一次
# assert_called_with(...) 验证最近一次调用的参数
# assert_called_once_with(...) 验证恰好调用一次，并且参数符合预期
# call_count 获取调用次数
# call_args 获取最近一次调用的参数
# call_args_list 获取所有调用的参数记录

# 模拟返回值
def test_result() -> None:
    client = Mock()
    # 规则：哪个方法需要返回指定结果，就设置哪个方法的 return_value。
    client.geqt.return_value = {"id": 1, "name": "tom", "age": 18}

    result = load_user2(client, 1)
    print("result ", result)

    client.geqt.assert_called_once_with(
        id=1,
    )
    assert result == {"id": 1, "name": "tom", "age": 18}
    pass


def test_load_user_error():
    client = Mock()
    # 六、规则 5：需要模拟异常时，使用 side_effect
    client.get.side_effect = TimeoutError("请求超时")

    # try:
    #     load_user(client, 1)
    #     assert False, "应该抛出 TimeoutError"
    # except TimeoutError as exc:
    #     assert str(exc) == "请求超时"

    with pytest.raises(TimeoutError, match="请求超时"):
        load_user(client, 1)
    pass

# 七、规则 6：Mock 默认过于宽松，重要依赖可以使用 spec
class MockAPIClient:
    def get(self, id):
        pass

def test_mock_api_client():
    # 通过 spec 来设置的钱mock的是哪一个对象，在mock调用的方法必须是 spec 指定的 MockAPIClient 类里面存在的方法，可以限制合法方法的调用，防止方法被乱写
    client = Mock(spec=MockAPIClient)
    client.get.return_value = {"id": 1, "name": "tom", "age": 18}

    result = load_user(client, 1)
    print("result ", result)

    client.get.assert_called_once_with(
        id=1,
    )
    assert result == {"id": 1, "name": "tom", "age": 18}

    pass


BASE_URL = "https://jsonplaceholder.typicode.com"


def load_user_from_http(client: httpx.Client, user_id: int) -> dict:
    """通过 HTTP 请求获取用户信息。"""
    response = client.get(
        url=f"{BASE_URL}/users/{user_id}",
        timeout=10.0,
    )

    # HTTP 状态码不是 2xx 时抛出异常
    response.raise_for_status()

    # 将 JSON 响应转换为 Python 字典
    return response.json()

def test_load_user_from_http():
    # 真实 HTTP 请求
    with httpx.Client() as client:
        user = load_user_from_http(client, 1)
        print(user)

def test_load_user_from_http_success():
    # 使用mock来模拟http的调用，实际不调用
    client = Mock()
    user_data = {'id': 1, 'name': 'Leanne111 Graham', 'username': 'Bret', 'email': 'Sincere@april.biz'}

    # 模拟 HTTP 响应对象
    response = Mock()
    response.json.return_value = user_data

    # 模拟 client.get() 返回 response
    client.get.return_value = response

    user_id = 1
    res = load_user_from_http(client, user_id)
    print(res)

    client.get.assert_called_once_with(
        url=f"{BASE_URL}/users/{user_id}",
        timeout=10.0,
    )

    pass


def test_load_user_from_http_failed():
    # 测试 http 返回404时是否抛出异常
    client = Mock()
    url = f"{BASE_URL}/users/{999}"
    request = httpx.Request("GET", url)
    response = httpx.Response(
        status_code=404,
        request=request,
    )
    client.get.return_value = response

    # try:
    #     res = load_user_from_http(client, 999)
    #     print("res -> ", res)
    # except Exception as e:
    #     print(str(e))

    with pytest.raises(httpx.HTTPError):
        res = load_user_from_http(client, 999)
        print("res -> ", res)

    pass


# 临时文件读写测试
def write_data(output, data):

    print("output : ", output)
    print("data : ", data)

    with open(output, "w") as f:
        json.dump(data, f)
    pass


def test_tmp_file(tmp_path) -> None:
    output = tmp_path / "output.json"

    write_data(output, {"id": 1, "name": "tom", "age": 18})

    assert output.exists()
    assert json.loads(output.read_text()) == {"id": 1, "name": "tom", "age": 18}

    pass


# 测试异步函数：
import asyncio

async def fetch_value() -> int:
    await asyncio.sleep(1)
    return 42

async def test_fetch_value() -> None:
    print(time.time())
    assert await fetch_value() == 42
    print(time.time())





