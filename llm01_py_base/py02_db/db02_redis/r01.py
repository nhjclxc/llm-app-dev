import asyncio
import json
import time

import pytest
import redis
from redis import WatchError


def open_redis(host:str = "localhost", port:int = 6379, db:int=0) -> redis.Redis:
    # 直接创建redis实例
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        # `decode_responses=True` 让 Redis 返回 Python 字符串；不设置时通常返回 `bytes`。二进制文件、压缩内容等场景应保留 bytes。
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=5,
    )

def create_redis(url:str = "redis://localhost:6379/0") -> redis.Redis:
    # 使用连接池来创建redis
    pool = redis.ConnectionPool.from_url(
        url=url,
        decode_responses=True,
        max_connections=10,
    )
    return redis.Redis(connection_pool=pool)


if __name__ == '__main__':
    client = open_redis()
    client = create_redis()
    print(client)
    print(client.ping())

    print(client.set("key-py", "value", ex=3))
    print(client.get("key-py"))
    time.sleep(4)
    print(client.get("key-py"))

    try:
        client.ping()
    except redis.TimeoutError:
        print("redis timeout")
    except redis.exceptions.ConnectionError:
        print("redis connection error")
    except redis.RedisError as e:
        print("redis error")


    client.close()


    # Key 命名规范
    # 推荐使用冒号分隔的业务层级：
    # Key 应包含业务域和唯一 ID，缓存、锁、验证码和临时幂等记录必须设计 TTL。


    pass


def test_ttl() -> None:
    client = create_redis()
    client.set("key-py", "value")
    assert client.get("key-py") == "value"
    client.delete("key-py")


    # # TTL 返回值：`>=0` 表示剩余秒数，`-1` 表示没有过期时间，`-2` 表示 Key 不存在。

    client.set("code:13800000000", "123456", ex=300)
    print(client.ttl("code:13800000000"))
    time.sleep(2)
    print(client.ttl("code:13800000000"))

    pass

def test_nx() -> None:

    client = create_redis()
    # 如果 nx 设置为 True，则仅当键名不存在时才将其值设置为值。
    client.set("key-py123", "value123", nx=True)
    # 如果 xx 设置为 True，则仅当键名处的值已存在时才将其设置为该值。
    client.set("key-py123", "value123321", xx=True)

    pass

def test_incr() -> None:

    client = create_redis()

    # 实现计数器, 设置初始值为0
    client.set("count-py", 0)
    for i in range(5):
        # Redis 的 `INCR` 是原子的，但“读取、判断、再写入”不一定原子。
        client.incr("count-py")
        time.sleep(1)
        print("result:", client.get("count-py"))
    # 直接自增3
    # 是增加后的库存值，而不是本次增加数量。
    res1 = client.incrby("count-py", 3)
    print("result:", client.get("count-py"))
    print("res1:", res1)

    pass


def test_json_str() -> None:
    client = create_redis()
    id = 1
    data = {"id": id, "key": "value", "key2": "value2", "version": int(time.time()*1000)}
    client.set(f"user:{id}", json.dumps(data, ensure_ascii=False))

    raw = client.get(f"user:{id}")
    print(raw)
    if raw is not None:
        user = json.loads(raw)
        print(user)

    pass


def test_hash() -> None:
    client = create_redis()

    client.hset("hash-user-1", mapping={"name": "tom", "age": 19})
    assert client.hget("hash-user-1", "name") == "tom"
    print(client.hgetall("hash-user-1"))

    client.hincrby("hash-user-1", "count")
    print(client.hget("hash-user-1", "count"))

    assert int(client.hget("hash-user-1", "age")) == 19
    client.hdel("hash-user-1", "age")
    assert client.hget("hash-user-1", "age") is None


    pass


def test_list_queue() -> None:
    # 列表和队列
    client = create_redis()
    client.lpush("queue-py",1, 2, 3, 4, 5)
    client.rpush("queue-py",111, 222, 333)

    print(client.lrange("queue-py", 0, -1))

    for _ in range(3):
        # # `BLPOP` 取出后消息就从队列删除。消费者处理中崩溃可能导致任务丢失；需要确认和恢复时学习 Streams 或专门任务队列。
        item = client.blpop("queue-py", timeout=5)
        if item is not None:
            queue_name, job_id = item
            print(queue_name, job_id)

    client.close()

def test_set() -> None:
    client = create_redis()

    # Set 无序，不能依赖成员返回顺序。

    client.sadd("set-py", 1,2,3,4,5)
    # smismember 检查元素是否在某个set里面
    # 返回 `values` 中的每个值是否属于集合 `name`，以 `int` 列表的形式按 `values` 的顺序返回： - 如果该值属于集合，则返回 1。 - 如果该值不属于集合，或者键不存在，则返回 0。
    print(client.smismember("set-py", (2,3,12)))

    client.sadd("course:python", "tom", "jerry")
    client.sadd("course:redis", "tom", "alice")
    # 交
    print(client.sinter("course:python", "course:redis"))
    # 并
    print(client.sunion("course:python", "course:redis"))
    # 差
    print(client.sdiff("course:python", "course:redis"))

    client.close()
    pass

def test_zset() -> None:
    client = create_redis()

    client.zadd("rank:score", {"tom": 95, "jerry": 88, "alice": 99, "jack": 98})

    print(client.zscore("rank:score", "tom"))
    # score从高到低，排名从0开始
    print(client.zrevrank("rank:score", "tom"))
    print(client.zrevrange("rank:score", 0, -1))
    print(client.zrevrange("rank:score", 0, -1, withscores=True))

    print(client.zrangebyscore("rank:score", 90, 100, withscores=True))
    print(client.zrevrangebyscore("rank:score", 100, 90, withscores=True))

    client.close()
    pass



def test_scan_key() -> None:
    client = create_redis()
    # 扫描key，
    # 不要使用key*，使用key*会被阻塞
    keys = client.scan_iter("diattest:*")
    for key in keys:
        print(key)

    # 删除keys
    client.delete(*keys)

    # 大规模删除可以使用`UNLINK`，让内存释放异步执行：
    # client.unlink(*keys)

    client.close()
    pass


def test_pipeline() -> None:
    client = create_redis()

    # Pipeline 和事务
    # 使用 pipeline 可以减少网络通信

    # transaction=False 表示非事务pipe，仍然可以批量发送命令，但是不保证命令的事务
    with client.pipeline(transaction=False) as pipe:
        pipe.set("pipe1", "value1")
        pipe.set("pipe2", "value2")
        pipe.set("pipe3", "value3")
        pipe.set("pipe13", "value13")
        pipe.incr("pipe-count")
        # 整体提交更新
        result = pipe.execute()
        print(result)

    with client.pipeline() as pipe:
        # - multi()：开启事务模式，后续命令先进入事务队列，不立即执行。
        pipe.multi()
        pipe.set("order:1:status", "paid")
        pipe.set("order:1:paid_at", "2026-10-10T10:00:00+08:00")
        # - execute()：提交并执行事务队列中的命令，Redis按顺序执行这些命令。
        pipe.execute()

    client.close()
    pass



def deduct_stock(client: redis.Redis, key: str, amount: int) -> bool:
    # deduct = client.register_script("""
    # local stock = tonumber(redis.call('GET', KEYS[1]) or '0')
    # local amount = tonumber(ARGV[1])
    # if stock < amount then return 0 end
    # redis.call('DECRBY', KEYS[1], amount)
    # return 1
    # """)

    with client.pipeline() as pipe:
        for _ in range(5):
            try:
                pipe.watch(key)
                current = int(pipe.get(key) or 0)
                if current < amount:
                    pipe.unwatch()
                    return False
                pipe.multi()
                pipe.decrby(key, amount)
                pipe.execute()
                return True
            except WatchError:
                continue
    return False

def test_atomic_opt() -> None:
    client = create_redis()
    key = "goods:1001"
    client.set(key, 5, nx=True)
    amount = 3
    amount3_res = deduct_stock(client, key, amount)
    print(f"{key}:{amount} -> {amount3_res}")

    client.close()
    pass


def update_goods(client: redis.Redis, key: str, amount: int) -> bool:
    # 商品的key和对应锁的key不能是同一个，否则永远都是获取锁失败
    lock_key = f"lock:{key}"
    lock = client.lock(lock_key, timeout=5, blocking_timeout=3)
    if not lock.acquire():
        raise RuntimeError("获取锁失败")

    try:
        return deduct_stock(client, key, amount)
    finally:
        # 最终必须释放锁
        lock.release()


# pip install pytest-asyncio
# uv add pytest-asyncio
@pytest.mark.asyncio
async def test_lock() -> None:
    client = create_redis()

    key = "goods:1002"
    client.set(key, 10, xx=True)

    # 分布式锁
    # assert update_goods(client, key, 3) is True
    # assert update_goods(client, key, 3) is False

    # 不改同步函数，要用 asyncio 并发调用它，最直接的办法就是 asyncio.to_thread()。

    concurrency = 3
    tasks = [
        asyncio.to_thread(update_goods, client, key, 3)
        for _ in range(concurrency)
    ]

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    success_count = sum(
        not isinstance(result, BaseException)
        for result in results
    )
    failure_count = concurrency - success_count

    print(f"并发任务数: {concurrency}")
    print(f"成功数: {success_count}")
    print(f"失败数: {failure_count}")
    print(f"最终库存: {client.get(key)}")

    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            print(f"任务 {i} 失败: {result!r}")

    client.close()


def test_pub_sub():
    client = create_redis()
    pubsub = client.pubsub()

    try:
        # 1. 先订阅频道
        pubsub.subscribe("event:user")

        # 2. 等待 Redis 确认订阅成功
        message = pubsub.get_message(timeout=2)
        print("订阅确认：", message)

        # 3. 发布消息
        data = json.dumps({
            "name": "zs",
            "age": 18,
        })
        res = client.publish("event:user", data)
        print("发布时的订阅者数量：", res)

        # 4. 获取消息
        message = pubsub.get_message(
            ignore_subscribe_messages=True,
            timeout=2,
        )
        print("收到消息：", message)

        if message is not None:
            print("消息数据：", message["data"])
            print("解析结果：", json.loads(message["data"]))

    finally:
        pubsub.close()
        client.close()

def test_pub():
    client = create_redis()

    # 发布
    event_key = "event:goods"
    res = client.publish(event_key, json.dumps({"goods_name": "tx", "size": 11}))
    print("发布时的订阅者数量：", res)

    client.close()

def test_sub():
    client = create_redis()
    print()

    # 订阅
    pubsub = client.pubsub()
    event_key = "event:goods"

    try:
        # 订阅频道
        pubsub.subscribe(event_key)
        # 获取订阅成功消息
        message = pubsub.get_message(timeout=1)
        print("订阅成功消息：", message)
        if message is None or message["type"] != "subscribe":
            raise RuntimeError("未收到订阅确认消息")

        # 持续监听频道
        print("开始持续监听：", event_key)
        for message in pubsub.listen():
            print("收到消息：", message)

            if message["type"] == "message":
                print("消息内容：", message["data"])

        pass
    finally:
        # 关闭资源
        pubsub.close()
        client.close()

    pass


def test_stream() -> None:
    client = create_redis()



    client.close()


