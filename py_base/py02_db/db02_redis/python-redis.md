# Python 使用 Redis 教程

本文使用 `redis-py` 学习 Redis，覆盖连接、数据类型、过期、事务、Pipeline、Lua、缓存、锁、消息、异步客户端、测试和常见问题。


pip install py-redis
uv add py-redis
jdbc:redis://localhost:6379/0


## 1. 环境准备

启动 Redis：

```bash
docker run --name study-redis -p 6379:6379 -d redis:7
redis-cli ping
```

安装客户端：

```bash
python -m pip install redis
```

`redis-py` 同时提供同步客户端和 `redis.asyncio` 异步客户端。

## 2. 第一个程序

```python
import redis


client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
    socket_connect_timeout=3,
    socket_timeout=5,
)

print(client.ping())
client.set("user:1:name", "Tom")
print(client.get("user:1:name"))
client.delete("user:1:name")
client.close()
```

`decode_responses=True` 让 Redis 返回 Python 字符串；不设置时通常返回 `bytes`。二进制文件、压缩内容等场景应保留 bytes。

## 3. 配置、连接池和异常

生产代码不要把密码写入源码：

```python
import os
import redis


def create_client() -> redis.Redis:
    return redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=5,
        health_check_interval=30,
    )
```

`redis.Redis` 默认使用连接池。也可以显式配置：

```python
pool = redis.ConnectionPool.from_url(
    "redis://localhost:6379/0",
    max_connections=20,
    decode_responses=True,
)
client = redis.Redis(connection_pool=pool)
```

异常处理：

```python
from redis.exceptions import RedisError, TimeoutError


try:
    client.ping()
except TimeoutError:
    # 可以执行降级逻辑
    pass
except RedisError as exc:
    raise RuntimeError("Redis 操作失败") from exc
```

应用退出时关闭客户端：同步使用 `client.close()`，异步使用 `await client.aclose()`。

## 4. Key 命名规范

推荐使用冒号分隔的业务层级：

```text
user:1001:profile
cache:product:1001
session:token:abc
lock:order:1001
rate_limit:login:ip:1.2.3.4
```

Key 应包含业务域和唯一 ID，不使用空格，不直接使用未经处理的用户输入。缓存、锁、验证码和临时幂等记录必须设计 TTL。

## 5. String：字符串、数字和 JSON

```python
client.set("name", "Tom")
assert client.get("name") == "Tom"
client.delete("name")

client.set("code:13800000000", "123456", ex=300)
print(client.ttl("code:13800000000"))
```

TTL 返回值：`>=0` 表示剩余秒数，`-1` 表示没有过期时间，`-2` 表示 Key 不存在。

只在不存在时写入：

```python
client.set("once:key", "value", nx=True)
client.set("existing:key", "new", xx=True)
```

计数器：

```python
client.set("page:view_count", 0)
client.incr("page:view_count")
client.incrby("page:view_count", 10)
```

Redis 的 `INCR` 是原子的，但“读取、判断、再写入”不一定原子。

JSON：

```python
import json

user = {"id": 1, "name": "Tom", "roles": ["admin"]}
client.set("user:1", json.dumps(user, ensure_ascii=False), ex=3600)
raw = client.get("user:1")
if raw is not None:
    user = json.loads(raw)
```

JSON 建议保存 `version` 字段，便于以后升级格式。

## 6. Hash：对象字段

```python
client.hset("user:1", mapping={"name": "Tom", "age": "18"})
assert client.hget("user:1", "name") == "Tom"
print(client.hgetall("user:1"))
client.hincrby("user:1", "login_count", 1)
client.hdel("user:1", "age")
```

Hash 适合用户资料、任务状态和设备状态。字段值通常仍是字符串，需要在 Python 中转换成 `int`、`float` 或布尔值。

## 7. List：列表和简单队列

```python
client.rpush("jobs", "job-1", "job-2")
client.lpush("jobs", "job-0")
print(client.lrange("jobs", 0, -1))

item = client.blpop("jobs", timeout=5)
if item is not None:
    queue_name, job_id = item
```

`BLPOP` 取出后消息就从队列删除。消费者处理中崩溃可能导致任务丢失；需要确认和恢复时学习 Streams 或专门任务队列。

## 8. Set：去重和集合运算

```python
client.sadd("user:1:tags", "python", "redis", "backend")
assert client.sismember("user:1:tags", "redis")
client.srem("user:1:tags", "backend")

client.sadd("course:python", "tom", "jerry")
client.sadd("course:redis", "tom", "alice")
client.sinter("course:python", "course:redis")
client.sunion("course:python", "course:redis")
client.sdiff("course:python", "course:redis")
```

Set 无序，不能依赖成员返回顺序。

## 9. Sorted Set：排行榜

```python
client.zadd("rank:score", {"tom": 95, "jerry": 88, "alice": 99})
print(client.zrevrange("rank:score", 0, 2, withscores=True))
print(client.zrevrank("rank:score", "tom"))
print(client.zscore("rank:score", "tom"))
client.zrangebyscore("rank:score", min=90, max=100, withscores=True)
```

相同分数时会按 member 字典序排序。生产排行榜需要考虑分页、分数更新和用户删除。

## 10. Key 扫描

不要在生产环境使用 `KEYS *`，它可能阻塞 Redis。使用游标扫描：

```python
keys = list(client.scan_iter(match="cache:product:*", count=100))
if keys:
    client.delete(*keys)
```

大规模删除可以使用 `UNLINK`，让内存释放异步执行：

```python
if keys:
    client.unlink(*keys)
```

## 11. Pipeline 和事务

Pipeline 减少网络往返：

```python
with client.pipeline(transaction=False) as pipe:
    pipe.set("a", "1")
    pipe.set("b", "2")
    pipe.incr("counter")
    results = pipe.execute()
```

默认 pipeline 会使用事务；只需要批量发送时设置 `transaction=False`。Pipeline 减少往返，不等于所有命令都具备业务原子性。

事务示例：

```python
with client.pipeline() as pipe:
    pipe.multi()
    pipe.set("order:1:status", "paid")
    pipe.set("order:1:paid_at", "2026-10-10T10:00:00+08:00")
    pipe.execute()
```

Redis 事务不会像关系型数据库一样自动回滚已经执行的命令。

## 12. WATCH 和 Lua

### 12.1 WATCH 乐观锁

```python
from redis.exceptions import WatchError


def deduct_stock(client: redis.Redis, key: str, amount: int) -> bool:
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
```

并发冲突过高时 `WatchError` 会频繁发生，应设置重试上限或改用 Lua。

### 12.2 Lua 原子扣库存

```python
deduct = client.register_script("""
local stock = tonumber(redis.call('GET', KEYS[1]) or '0')
local amount = tonumber(ARGV[1])
if stock < amount then return 0 end
redis.call('DECRBY', KEYS[1], amount)
return 1
""")

success = deduct(keys=["stock:book"], args=[2]) == 1
```

Lua 执行期间会阻塞 Redis，脚本必须短小、确定，不能进行大量循环。

## 13. 缓存 Cache-Aside

```python
import json


def get_user(client: redis.Redis, user_id: int) -> dict | None:
    key = f"cache:user:{user_id}"
    cached = client.get(key)
    if cached is not None:
        return None if cached == "null" else json.loads(cached)

    user = load_user_from_database(user_id)
    if user is None:
        # 防止缓存穿透：不存在结果使用短 TTL
        client.set(key, "null", ex=60)
        return None

    client.set(key, json.dumps(user, ensure_ascii=False), ex=600)
    return user
```

更新数据库后通常删除缓存：

```python
save_user_to_database(user_id, data)
client.delete(f"cache:user:{user_id}")
```

三种缓存问题：

- 穿透：大量请求不存在的数据；使用空值短 TTL、参数校验或布隆过滤器；
- 击穿：热点 key 同时过期；使用互斥锁、逻辑过期或预热；
- 雪崩：大量 key 同时过期；使用随机 TTL、分批过期和降级。

随机 TTL：

```python
import random

client.set(key, value, ex=600 + random.randint(0, 60))
```

## 14. 分布式锁

```python
lock = client.lock("lock:order:1001", timeout=10, blocking_timeout=3)
if not lock.acquire():
    raise RuntimeError("获取锁失败")
try:
    process_order(1001)
finally:
    lock.release()
```

必须设置 TTL，并在 `finally` 释放。业务执行时间超过 TTL 时需要续期或重新设计。不要用简单的 `SET` 后 `DEL`，否则可能误删其他请求获得的新锁。

## 15. Pub/Sub 和 Streams

发布订阅：

```python
client.publish("events:user", '{"type":"login","user_id":1}')

pubsub = client.pubsub()
pubsub.subscribe("events:user")
for message in pubsub.listen():
    if message["type"] == "message":
        print(message["data"])
```

Pub/Sub 不保存历史，订阅者离线期间会丢消息。需要确认、重试和恢复时使用 Streams：

```python
message_id = client.xadd(
    "stream:orders",
    {"order_id": "1001", "event": "created"},
)
client.xgroup_create(
    "stream:orders", "order-workers", id="0", mkstream=True,
)
messages = client.xreadgroup(
    "order-workers", "worker-1", {"stream:orders": ">"},
    count=10, block=5000,
)
client.xack("stream:orders", "order-workers", message_id)
```

Streams 仍然可能重复投递，业务处理必须幂等。

## 16. 其他数据结构

HyperLogLog：近似去重计数，不能用于精确账务：

```python
client.pfadd("uv:2026-10-10", "user-1", "user-2")
count = client.pfcount("uv:2026-10-10")
```

Bitmap：签到和布尔标记：

```python
client.setbit("sign:2026-10", 1001, 1)
signed = client.getbit("sign:2026-10", 1001)
```

Geo：附近地点查询：

```python
client.geoadd("shops", (121.4737, 31.2304, "shop:1"))
client.geosearch("shops", longitude=121.4737, latitude=31.2304,
                 radius=5, unit="km")
```

## 17. 异步 Redis

```python
import redis.asyncio as redis


async def main() -> None:
    client = redis.Redis.from_url(
        "redis://localhost:6379/0",
        decode_responses=True,
    )
    try:
        await client.set("async:key", "value", ex=60)
        print(await client.get("async:key"))
    finally:
        await client.aclose()
```

异步 pipeline：

```python
async with client.pipeline(transaction=False) as pipe:
    pipe.set("a", "1")
    pipe.set("b", "2")
    await pipe.execute()
```

异步服务中不要把同步 Redis 客户端放入高并发请求路径，否则同步网络 I/O 会阻塞事件循环。

## 18. FastAPI 生命周期

```python
from contextlib import asynccontextmanager
import redis.asyncio as redis
from fastapi import FastAPI, Request


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = redis.Redis.from_url(
        "redis://localhost:6379/0", decode_responses=True
    )
    await app.state.redis.ping()
    yield
    await app.state.redis.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/cache/{key}")
async def read_cache(key: str, request: Request):
    value = await request.app.state.redis.get(key)
    return {"key": key, "value": value}
```

不要在每个请求里创建 Redis 客户端。生产服务还应补充超时、健康检查、错误日志和 Redis 不可用时的降级策略。

## 19. 测试 Redis 代码

### 19.1 单元测试：Mock

```python
from unittest.mock import Mock


def test_cache_hit():
    client = Mock()
    client.get.return_value = '{"id": 1, "name": "Tom"}'
    result = get_user(client, 1)
    assert result["name"] == "Tom"
    client.get.assert_called_once_with("cache:user:1")
```

### 19.2 集成测试

TTL、事务、Lua、Streams 应使用独立 Redis 实例测试。可用 Docker Compose、Testcontainers 或 CI Redis 服务。测试 Key 使用唯一前缀，并在 fixture 的 `finally` 中清理，不要清空共享 Redis。

## 20. Redis 常见错误

1. 缓存、验证码和锁没有 TTL，导致内存持续增长；
2. 生产环境使用 `KEYS *`，造成阻塞；
3. 把 Pub/Sub 当可靠队列，导致离线消息丢失；
4. 误以为事务会自动回滚；
5. 反复创建 Redis 客户端，浪费连接；
6. 异步项目中调用同步客户端，阻塞事件循环；
7. 用字符串拼接保存复杂对象，无法稳定解析；
8. 缓存更新和数据库更新顺序不清晰；
9. 锁没有过期时间或直接删除其他请求的锁；
10. Redis 连接没有设置超时，故障时拖垮请求线程。

## 21. 安全和生产注意事项

- Redis 不要直接暴露公网；
- 使用认证、TLS 和网络访问控制；
- 日志中不输出密码和完整连接 URL；
- 设置最大内存和淘汰策略；
- 根据数据重要性选择 RDB、AOF 或高可用方案；
- 监控内存、命中率、连接数、慢查询和阻塞客户端；
- 不把 Redis 作为唯一永久数据存储，除非设计了持久化和恢复；
- 生产、测试使用不同实例或明确隔离的数据库。

## 22. 学习顺序

### 阶段一：基础数据类型

连接、String、Hash、List、Set、Sorted Set、TTL、计数器和 JSON。

### 阶段二：性能和原子性

Pipeline、事务、WATCH、Lua、SCAN 和分布式锁。

### 阶段三：消息和异步

Pub/Sub、Streams、消费者组、`redis.asyncio` 和 FastAPI 生命周期。

### 阶段四：综合项目

实现订单服务：Hash 保存状态、String 保存缓存、Sorted Set 做排行、Stream 记录事件、Lua 扣库存、Lock 防重复处理、pytest 测试并发和重复消息。

## 23. 学习完成标准

- [ ] 能创建带连接池、超时和环境配置的客户端；
- [ ] 能正确使用 String、Hash、List、Set、Sorted Set；
- [ ] 能设置和检查 TTL；
- [ ] 能区分 Pipeline、事务、WATCH 和 Lua；
- [ ] 能解释 Pub/Sub 与 Streams 的差异；
- [ ] 能安全实现基本分布式锁；
- [ ] 能正确关闭异步客户端；
- [ ] 能在 FastAPI 生命周期中管理 Redis；
- [ ] 能用 Mock 和独立 Redis 实例测试；
- [ ] 能设计缓存穿透、击穿和雪崩的处理方案；
- [ ] 能说明 Redis 持久化、淘汰和内存风险；
- [ ] 能设计 Redis 不可用时的降级策略。

