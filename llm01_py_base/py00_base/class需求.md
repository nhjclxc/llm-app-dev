# Python Class 面向对象综合练习题

## 1. 练习目标

通过本套练习系统掌握 Python 面向对象编程，重点包括：

- 类与对象；
- 实例属性、类属性和局部变量；
- 实例方法、类方法和静态方法；
- 构造方法与对象生命周期；
- 封装、属性校验和访问控制；
- 继承、方法重写和 `super()`；
- 多态、鸭子类型和协议；
- 抽象类与抽象方法；
- 组合、聚合与继承的选择；
- `property`、描述符和数据校验；
- 魔术方法；
- `dataclass`、`__slots__` 和不可变对象；
- 多继承、Mixin 和 MRO；
- 类型标注、泛型和 `Protocol`；
- 异常设计、资源管理和测试；
- 面向对象设计原则与可维护的项目结构。

本套题不要求一开始追求复杂架构。每一道题都应先写出最简单的可运行版本，再根据后续要求逐步重构。

## 2. 总体要求

### 2.1 推荐代码位置

建议在 `py_base/py00_base/` 下创建：

```text
base33_class.py
```

如果题目代码较多，可以拆成：
base33_class_base.py
```text
py_base/py00_base/class_practice/
  __init__.py
  base33_class01_base.py
  base33_class02_encapsulation.py
  base33_class03_inheritance.py
  base33_class04_polymorphism.py
  base33_class05_abstract.py
  base33_class06_magic_methods.py
  base33_class07_design.py
  base33_class08_project.py
  tests/
```

### 2.2 代码要求

- 使用 Python 3.11 或以上版本。
- 所有公开类、方法和函数添加类型标注。
- 类名使用 `PascalCase`，方法和属性使用 `snake_case`。
- 每个阶段都要有可运行示例。
- 不使用全局可变变量保存核心业务状态。
- 不用异常捕获掩盖设计问题。
- 不为了“使用继承”而强行继承；能用组合解决时，必须比较两种方案。
- 每个阶段至少写 3 个断言或测试用例。
- 代码中不得出现真实密码、API Key、Cookie 或私人数据。

### 2.3 每个阶段都要提交的内容

每完成一个阶段，都要记录：

1. 本阶段新增的类和职责。
2. 对象之间的关系：继承、组合、依赖或关联。
3. 一个最容易写错的地方。
4. 一个失败案例和修复方式。
5. 为什么当前设计比简单字典或函数更合适。

## 3. 阶段 1：类与对象基础

### 3.1 练习一：学生对象

定义 `Student` 类：

```python
class Student:
    ...
```

属性要求：

- `name`：姓名；
- `student_id`：学号；
- `scores`：课程成绩字典，例如 `{"python": 90, "english": 80}`；
- `created_at`：创建时间。

方法要求：

- `add_score(course, score)`：增加或修改成绩；
- `get_score(course)`：获取指定课程成绩；
- `average_score()`：计算平均分；
- `is_passed()`：判断平均分是否及格；
- `to_dict()`：转换为普通字典；
- `__str__()`：输出适合用户阅读的信息；
- `__repr__()`：输出适合调试的信息。

校验要求：

- 姓名不能为空；
- 学号不能为空；
- 成绩必须在 0 到 100 之间；
- 课程名不能为空；
- 查询不存在的课程时要有明确行为，不允许静默返回错误数据。

### 3.2 练习二：理解实例属性和类属性

给 `Student` 增加类属性：

```python
school_name = "Unknown School"
student_count = 0
```

要求：

- 每创建一个学生，`student_count` 增加 1；
- 能通过类名修改学校名称；
- 比较 `Student.school_name` 和 `student.school_name` 的查找关系；
- 演示实例属性遮蔽类属性的情况；
- 不允许把每个学生独有的成绩放到类属性中。

### 3.3 练习三：方法分类

为 `Student` 增加以下方法：

- 实例方法：`average_score()`；
- 类方法：`from_dict(data)`，从字典创建学生；
- 静态方法：`validate_score(score)`，校验成绩是否合法。

要求在注释中解释：

- 实例方法为什么需要 `self`；
- 类方法为什么使用 `cls`；
- 静态方法为什么不需要 `self` 或 `cls`；
- `from_dict()` 为什么适合写成类方法而不是普通函数。

### 3.4 阶段验收

- [ ] 能创建多个互不影响的 `Student` 对象。
- [ ] 能正确区分实例属性和类属性。
- [ ] 能使用实例方法、类方法和静态方法。
- [ ] `str(student)` 和 `repr(student)` 有不同用途。
- [ ] 非法成绩和空字段会被拒绝。
- [ ] 测试至少覆盖正常学生、无成绩学生和非法成绩。

## 4. 阶段 2：封装与属性控制

### 4.1 练习四：银行账户

定义 `BankAccount` 类：

属性：

- `account_id`：账户编号；
- `owner`：账户持有人；
- `_balance`：余额，约定为受保护属性；
- `__transaction_history`：交易记录，使用名称改写隐藏；
- `is_frozen`：账户是否冻结。

方法：

- `deposit(amount)`：存款；
- `withdraw(amount)`：取款；
- `freeze()`：冻结账户；
- `unfreeze()`：解冻账户；
- `get_transaction_history()`：返回交易记录副本；
- `transfer_to(other, amount)`：转账；
- `balance` 属性：只读获取余额。

业务要求：

- 存款金额必须大于 0；
- 冻结账户不能存取款；
- 余额不足不能取款；
- 转账必须保证扣款和入账逻辑完整；
- 外部代码不能直接替换交易历史列表；
- 返回交易历史时不能暴露内部可变列表；
- 账户异常必须使用自定义异常。

### 4.2 自定义异常

至少定义：

```python
class AccountError(Exception):
    pass


class InsufficientBalanceError(AccountError):
    pass


class AccountFrozenError(AccountError):
    pass
```

要求：

- 调用方可以按异常类型处理不同错误；
- 异常消息包含账户编号和失败原因；
- 不要使用 `except Exception: pass`；
- 不要把所有业务错误都转换为普通字符串返回。

### 4.3 练习五：`property` 属性

定义 `Temperature` 类：

- 内部使用摄氏温度 `_celsius`；
- 提供 `celsius` 属性读写；
- 提供只读 `fahrenheit` 属性；
- 摄氏温度不能低于绝对零度；
- 修改摄氏温度后，华氏温度自动变化。

公式：

```text
fahrenheit = celsius * 9 / 5 + 32
```

要求：

- 不能额外保存一份可能过期的华氏温度；
- 通过属性访问时完成必要校验；
- 解释为什么 `fahrenheit` 适合只读属性。

### 4.4 练习六：只读视图与副本

为 `BankAccount` 增加：

- 只读账户编号；
- 只读当前余额；
- 交易记录只返回元组或深拷贝；
- 不允许通过返回值绕过账户校验修改内部状态。

### 4.5 阶段验收

- [ ] 能解释 `_name` 和 `__name` 的区别。
- [ ] 余额只能通过受控方法变化。
- [ ] 外部修改返回的交易记录不会破坏内部状态。
- [ ] `property` 可以完成读取、写入和派生属性计算。
- [ ] 业务错误使用了有意义的异常类型。
- [ ] 至少测试存款、取款、冻结、余额不足和转账失败。

## 5. 阶段 3：继承与方法重写

### 5.1 练习七：员工体系

定义基类 `Employee`：

属性：

- `employee_id`；
- `name`；
- `base_salary`。

方法：

- `calculate_salary()`：计算工资；
- `work()`：描述工作；
- `summary()`：返回员工摘要；
- `__str__()`：显示员工信息。

定义子类：

- `FullTimeEmployee`：固定工资 + 津贴；
- `PartTimeEmployee`：小时工资 × 工作小时；
- `CommissionEmployee`：基础工资 + 销售提成。

要求：

- 子类通过 `super().__init__()` 初始化父类属性；
- 子类重写 `calculate_salary()` 和 `work()`；
- 不要复制父类初始化逻辑；
- 父类公共方法仍然可以处理所有子类对象。

### 5.2 练习八：方法重写与 `super()`

给 `Employee.summary()` 增加统一摘要格式，子类只能补充自己的信息：

```text
员工编号：E001
姓名：Alice
员工类型：全职
本月工资：...
```

要求比较两种实现：

1. 子类完全重写 `summary()`；
2. 子类调用 `super().summary()` 后补充内容。

说明哪种方式更容易维护，以及父类摘要字段增加时两种方式的差异。

### 5.3 练习九：继承中的初始化顺序

设计三层继承：

```text
Person -> Employee -> Manager
```

要求：

- 每层类都有自己的属性；
- 每层类的 `__init__()` 都调用 `super()`；
- 打印初始化顺序；
- 增加一个方法重写，观察方法查找顺序；
- 解释为什么多层继承中推荐使用 `super()` 而不是直接调用父类名称。

### 5.4 阶段验收

- [ ] 能定义父类和多个子类。
- [ ] 子类能复用父类初始化和公共方法。
- [ ] 能使用 `super()` 调用父类逻辑。
- [ ] 能解释覆盖、扩展和重写的区别。
- [ ] 没有复制大量父类代码。
- [ ] 能画出员工类的继承关系图。

## 6. 阶段 4：多态与鸭子类型

### 6.1 练习十：统一支付接口

定义：

```python
def checkout(payment_method, amount: float) -> str:
    ...
```

实现以下支付类：

- `AlipayPayment`；
- `WechatPayment`；
- `BankCardPayment`；
- `MockPayment`。

每个类都实现：

```python
def pay(self, amount: float) -> str:
    ...
```

要求：

- `checkout()` 不使用 `isinstance()` 判断具体支付类型；
- `checkout()` 只依赖对象是否具有 `pay()` 方法；
- 新增支付方式时，不修改 `checkout()`；
- 支付金额必须经过统一校验；
- 支付失败使用统一异常或统一结果结构。

### 6.2 练习十一：鸭子类型

定义以下不继承同一个父类的对象：

- `EmailSender`，有 `send(message)`；
- `SmsSender`，有 `send(message)`；
- `ConsoleSender`，有 `send(message)`。

实现：

```python
def notify(sender, message: str) -> None:
    sender.send(message)
```

要求：

- 不强制要求发送器继承共同父类；
- 使用类型标注描述 `send()` 协议；
- 传入没有 `send()` 方法的对象时，错误要容易定位。

### 6.3 练习十二：多态报表

定义 `ReportExporter` 协议，支持：

- `CsvExporter`；
- `JsonExporter`；
- `MarkdownExporter`。

统一接口：

```python
def export_report(exporter, rows: list[dict]) -> str:
    ...
```

要求：

- 业务代码不关心具体导出格式；
- 每种导出器都能处理空列表；
- 字符串、数字、列表和嵌套字典至少各测试一次；
- 导出器不能修改输入数据。

### 6.4 阶段验收

- [ ] 能解释多态的实际价值，而不是只会背定义。
- [ ] 新增实现类时，调用方代码不需要大幅修改。
- [ ] 能使用鸭子类型设计接口。
- [ ] 能区分继承多态和协议/鸭子类型多态。
- [ ] 没有在核心流程中堆积大量 `if isinstance(...)`。

## 7. 阶段 5：抽象类与接口设计

### 7.1 练习十三：抽象存储接口

使用 `abc` 定义：

```python
from abc import ABC, abstractmethod


class Storage(ABC):
    @abstractmethod
    def save(self, key: str, value: str) -> None:
        ...

    @abstractmethod
    def get(self, key: str) -> str | None:
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        ...
```

实现：

- `MemoryStorage`：使用字典保存；
- `FileStorage`：使用 JSON 文件保存；
- `LoggingStorage`：记录每次操作，再委托给真正存储。

要求：

- 不能实例化未实现完整抽象方法的类；
- 上层业务只依赖 `Storage`；
- 替换存储实现时不修改上层业务；
- 文件存储要处理文件不存在、JSON 损坏和写入失败。

### 7.2 练习十四：抽象任务处理器

定义抽象类：

```python
class TaskHandler(ABC):
    @abstractmethod
    def can_handle(self, task: dict) -> bool:
        ...

    @abstractmethod
    def handle(self, task: dict) -> dict:
        ...
```

实现：

- `EmailTaskHandler`；
- `FileTaskHandler`；
- `NotificationTaskHandler`。

实现任务分发器：

```python
class TaskDispatcher:
    def register(self, handler: TaskHandler) -> None:
        ...

    def dispatch(self, task: dict) -> dict:
        ...
```

要求：

- 一个任务只交给合适的处理器；
- 没有匹配处理器时抛出明确异常；
- 支持动态注册新的处理器；
- 分发器不依赖具体处理器类名。

### 7.3 阶段验收

- [ ] 能使用 `ABC` 和 `abstractmethod`。
- [ ] 能设计小而稳定的抽象接口。
- [ ] 上层业务依赖抽象，而不是依赖具体实现。
- [ ] 能解释抽象类与普通父类的区别。
- [ ] 能判断何时应使用抽象类，何时使用 `Protocol`。

## 8. 阶段 6：组合、聚合与继承的选择

### 8.1 练习十五：订单系统

定义以下类：

- `Product`：商品；
- `OrderItem`：订单项；
- `Customer`：客户；
- `DiscountPolicy`：折扣策略；
- `Order`：订单；
- `OrderService`：订单服务。

关系要求：

- `Order` 组合多个 `OrderItem`；
- `OrderItem` 关联一个 `Product`；
- `Order` 关联一个 `Customer`；
- `Order` 依赖一个 `DiscountPolicy`；
- `OrderService` 负责订单流程，但不负责商品对象内部计算。

功能要求：

- 添加商品；
- 删除商品；
- 修改数量；
- 计算原价；
- 计算折扣；
- 计算最终价格；
- 取消订单；
- 完成订单；
- 输出订单摘要。

### 8.2 折扣策略实现

实现：

- `NoDiscount`；
- `PercentageDiscount`；
- `FullReductionDiscount`；
- `VipDiscount`。

要求：

- 订单不直接判断具体折扣类型；
- 更换折扣策略不修改 `Order` 核心计算代码；
- 折扣不能使最终价格小于 0；
- 商品单价、数量、折扣比例都要校验。

### 8.3 必须比较的方案

请分别思考以下设计：

1. 让 `Order` 继承 `Product` 是否合理？为什么？
2. 将 `DiscountPolicy` 作为 `Order` 的成员变量有什么好处？
3. 如果把所有业务写进 `OrderService`，会出现什么问题？
4. 什么时候“组合优于继承”？

### 8.4 阶段验收

- [ ] 能画出订单系统的对象关系图。
- [ ] 能区分 is-a 和 has-a 关系。
- [ ] 使用组合完成折扣策略切换。
- [ ] 订单、订单项和服务职责没有严重混杂。
- [ ] 能解释为什么没有使用不合理的继承。

## 9. 阶段 7：魔术方法

### 9.1 练习十六：可比较的购物车

定义 `ShoppingCart`：

实现以下魔术方法：

- `__init__()`；
- `__str__()`；
- `__repr__()`；
- `__len__()`：返回商品项数量；
- `__bool__()`：购物车是否有商品；
- `__contains__()`：判断是否包含某个商品编号；
- `__iter__()`：遍历订单项；
- `__getitem__()`：通过下标读取订单项；
- `__setitem__()`：修改订单项时完成校验；
- `__delitem__()`：删除订单项；
- `__eq__()`：比较两个购物车内容是否相同。

示例目标：

```python
cart = ShoppingCart()

if cart:
    print("购物车有商品")

print(len(cart))
for item in cart:
    print(item)
```

要求：

- 魔术方法行为符合 Python 常规直觉；
- 非法下标抛出合理异常；
- 迭代不会暴露不应修改的内部结构；
- `__repr__()` 可以帮助开发者定位对象状态。

### 9.2 练习十七：上下文管理器

定义 `Transaction` 类，实现：

- `__enter__()`；
- `__exit__()`。

使用目标：

```python
with account.transaction() as transaction:
    transaction.withdraw(100)
    transaction.deposit(50)
```

要求：

- 事务成功时提交；
- 事务内部发生异常时回滚；
- 退出时恢复必要状态；
- 明确说明 `__exit__()` 返回 `True` 和 `False` 的区别；
- 测试正常退出、业务异常和未知异常。

### 9.3 练习十八：可调用对象

定义 `RetryPolicy`：

```python
policy = RetryPolicy(max_attempts=3)
result = policy(callable_func)
```

要求：

- 实现 `__call__()`；
- 仅对指定异常重试；
- 记录实际尝试次数；
- 失败后保留最后一次异常上下文；
- 不允许无限重试；
- 重试间隔可以通过参数配置。

### 9.4 阶段验收

- [ ] 能实现常用容器相关魔术方法。
- [ ] 能使用上下文管理器管理资源或事务。
- [ ] 能实现可调用对象。
- [ ] 理解魔术方法不是越多越好，而是为了符合 Python 语义。
- [ ] 能解释 `__new__()` 和 `__init__()` 的职责差异。

## 10. 阶段 8：`dataclass`、不可变对象和 `__slots__`

### 10.1 练习十九：配置对象

使用 `dataclass` 定义 `AppConfig`：

字段：

- `app_name`；
- `host`；
- `port`；
- `debug`；
- `timeout`；
- `tags`，默认空列表但不能共享同一个可变默认值。

要求：

- 使用 `field(default_factory=list)`；
- 实现 `from_dict()`；
- 实现 `to_dict()`；
- 对端口、超时时间和应用名进行校验；
- 比较 `dataclass` 自动生成的 `__init__()`、`__repr__()` 和 `__eq__()`。

### 10.2 练习二十：不可变值对象

定义不可变的 `Money`：

- `amount`；
- `currency`。

要求：

- 创建后不能直接修改字段；
- 支持同币种相加；
- 不同币种相加抛出异常；
- 支持乘以正数；
- 金额不能为负；
- 正确实现 `__eq__()` 和 `__hash__()`，使其可作为字典 Key。

### 10.3 练习二十一：`__slots__`

定义两个用户类：

- `NormalUser`：普通对象；
- `SlotUser`：使用 `__slots__`。

观察并记录：

- 是否存在 `__dict__`；
- 能否动态增加新属性；
- 内存和属性访问的差异；
- 继承后 `__slots__` 的行为；
- 使用 `__slots__` 可能带来的限制。

不要把 `__slots__` 当作所有类的默认写法。必须结合对象数量、动态属性需求和框架兼容性判断。

### 10.4 阶段验收

- [ ] 能正确使用 `dataclass` 和 `default_factory`。
- [ ] 能创建不可变值对象。
- [ ] 能解释 `__slots__` 的优点和限制。
- [ ] 能区分实体对象、值对象和配置对象。

## 11. 阶段 9：描述符、类装饰器和高级属性控制

### 11.1 练习二十二：非空字符串描述符

实现 `NonEmptyString` 描述符：

```python
class User:
    name = NonEmptyString()
    email = NonEmptyString()
```

要求：

- 赋值时拒绝空字符串；
- 读取时返回对象自己的值；
- 每个字段的值不能互相覆盖；
- 描述符保存的值必须存储在实例中，而不是描述符自身；
- 支持错误提示中显示字段名。

### 11.2 练习二十三：范围校验描述符

实现 `RangeField(min_value, max_value)`：

```python
class Product:
    price = RangeField(0, None)
    stock = RangeField(0, 100000)
```

要求：

- 支持整数或浮点数；
- 不符合范围时抛出明确异常；
- 保存字段名；
- 比较描述符校验和 `property` 校验的优缺点。

### 11.3 练习二十四：注册类装饰器

实现一个插件注册器：

```python
@register_plugin("json")
class JsonPlugin:
    ...
```

要求：

- 自动保存插件名称和类对象；
- 名称重复时抛出异常；
- 可以通过名称创建插件实例；
- 提供 `list_plugins()`；
- 注册表不能在业务代码中被随意替换。

### 11.4 阶段验收

- [ ] 能解释描述符的 `__get__()`、`__set__()` 和 `__set_name__()`。
- [ ] 能说明描述符和 `property` 的适用差异。
- [ ] 能使用装饰器完成类注册。
- [ ] 能避免把实例数据错误地存储在类级别共享状态中。

## 12. 阶段 10：多继承、Mixin 和 MRO

### 12.1 练习二十五：日志与序列化 Mixin

定义：

- `LoggingMixin`：提供日志能力；
- `JsonSerializableMixin`：提供 `to_json()`；
- `TimestampMixin`：提供创建和更新时间；
- `Order`：组合这些能力。

示例：

```python
class Order(LoggingMixin, JsonSerializableMixin, TimestampMixin):
    ...
```

要求：

- Mixin 只提供一个清晰的小能力；
- Mixin 不应该代表完整业务实体；
- 使用 `super()` 保持协作式多继承；
- 打印 `Order.__mro__` 并解释顺序；
- 设计一个错误示例，说明 Mixin 之间命名冲突的风险。

### 12.2 练习二十六：钻石继承

构造：

```text
       A
      / \
     B   C
      \ /
       D
```

要求：

- 每个类的初始化方法都使用 `super()`；
- 记录每个类的初始化顺序；
- 打印 `D.__mro__`；
- 解释 Python 如何避免同一个父类被错误初始化多次；
- 删除某个 `super()`，观察结果变化并记录。

### 12.3 阶段验收

- [ ] 能解释 MRO 和 C3 线性化的基本结果。
- [ ] 能使用简单 Mixin，而不是滥用多继承。
- [ ] 能定位多继承中的方法冲突。
- [ ] 理解多继承中直接调用父类名称可能破坏协作式初始化。

## 13. 阶段 11：`Protocol`、类型标注和泛型

### 13.1 练习二十七：通知协议

定义：

```python
from typing import Protocol


class Notifier(Protocol):
    def send(self, target: str, message: str) -> bool:
        ...
```

实现：

- `EmailNotifier`；
- `ConsoleNotifier`；
- `MockNotifier`。

要求：

- 实现类不要求显式继承 `Notifier`；
- 函数参数使用 `Notifier` 类型；
- 用静态检查工具检查类型；
- 运行时不能只依赖类型标注完成参数校验。

### 13.2 练习二十八：泛型仓储

定义泛型接口：

```python
from typing import Generic, TypeVar


T = TypeVar("T")


class Repository(Generic[T]):
    ...
```

实现内存仓储，支持：

- `save(item)`；
- `get(item_id)`；
- `list_all()`；
- `delete(item_id)`。

要求：

- `UserRepository` 保存用户；
- `OrderRepository` 保存订单；
- 不同仓储不能混入错误类型；
- 记录泛型对代码可读性和静态检查的帮助。

### 13.3 阶段验收

- [ ] 能区分继承接口和结构化接口。
- [ ] 能用 `Protocol` 描述鸭子类型协议。
- [ ] 能用 `TypeVar` 和 `Generic` 创建简单泛型类。
- [ ] 类型标注不会被误认为运行时校验。

## 14. 阶段 12：综合项目——可扩展的大模型应用任务系统

这是本套练习的最终项目。要求把前面学习的类知识应用到一个简化的大模型应用后端中。

### 14.1 项目背景

实现一个“文本任务处理系统”，用户可以提交以下类型的任务：

- 摘要任务；
- 翻译任务；
- 分类任务；
- 问答任务；
- 批量文档处理任务。

系统不要求真实调用大模型，可以使用模拟模型客户端。重点是类之间的职责、扩展能力、异常处理和测试。

### 14.2 核心类要求

#### `Task`

字段：

- `task_id`；
- `task_type`；
- `content`；
- `status`；
- `created_at`；
- `result`；
- `error`。

状态至少包括：

```text
pending
running
success
failed
cancelled
```

要求：状态变更必须经过方法控制，不能允许外部随意赋值为任意字符串。

#### `ModelClient`

定义模型客户端抽象接口：

```python
class ModelClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...
```

实现：

- `MockModelClient`：返回模拟结果；
- `SlowModelClient`：模拟慢响应；
- `FailingModelClient`：模拟服务失败；
- `RecordingModelClient`：记录调用次数和输入。

#### `TaskHandler`

抽象任务处理器：

```python
class TaskHandler(ABC):
    @abstractmethod
    def supports(self, task_type: str) -> bool:
        ...

    @abstractmethod
    def handle(self, task: Task, client: ModelClient) -> str:
        ...
```

实现：

- `SummaryTaskHandler`；
- `TranslationTaskHandler`；
- `ClassificationTaskHandler`；
- `QuestionAnswerTaskHandler`。

#### `TaskRepository`

定义任务保存接口，并实现：

- `MemoryTaskRepository`；
- `JsonFileTaskRepository`。

#### `TaskDispatcher`

要求：

- 注册多个处理器；
- 根据任务类型找到处理器；
- 不依赖具体处理器类名；
- 找不到处理器时抛出自定义异常；
- 支持以后增加新的任务类型。

#### `TaskService`

负责：

- 创建任务；
- 查询任务；
- 执行任务；
- 更新任务状态；
- 保存任务结果；
- 处理异常；
- 取消未执行任务。

要求：

- 服务依赖抽象接口；
- 模型客户端、仓储和处理器都可以替换；
- 不能把所有逻辑写进一个巨大的类；
- 任务执行失败后保留错误原因；
- 不能让失败任务被错误标记为成功。

### 14.3 综合项目目录

建议：

```text
class_practice/
  domain.py              # Task、值对象、领域异常
  clients.py             # ModelClient 及实现
  handlers.py            # 各类任务处理器
  repositories.py        # 仓储抽象和实现
  dispatcher.py          # 任务分发
  services.py            # 应用服务
  main.py                # 演示程序
  tests/
    test_domain.py
    test_handlers.py
    test_service.py
    test_repository.py
```

### 14.4 综合项目最低功能

- [ ] 创建摘要任务并执行。
- [ ] 创建翻译任务并执行。
- [ ] 查询任务状态。
- [ ] 处理不存在的任务。
- [ ] 处理不支持的任务类型。
- [ ] 模拟模型调用失败。
- [ ] 保存任务到内存仓储。
- [ ] 保存任务到 JSON 文件仓储。
- [ ] 替换模型客户端而不修改任务服务。
- [ ] 替换任务处理器而不修改任务分发器核心逻辑。
- [ ] 输出任务执行日志。
- [ ] 为核心业务编写单元测试。

## 15. 面向对象设计思考题

请在 `class思考题.md` 中回答：

1. 类和对象分别是什么？
2. 实例属性、类属性和局部变量有什么区别？
3. 为什么可变默认值不应该直接作为类属性或函数默认参数？
4. `self` 和 `cls` 分别代表什么？
5. `@staticmethod` 什么时候有价值？什么时候普通函数更好？
6. `_name` 和 `__name` 的区别是什么？
7. Python 的私有属性是真正的 private 吗？
8. `property` 解决了什么问题？
9. 继承、组合和依赖分别表达什么关系？
10. 什么是方法重写？什么是方法重载？Python 是否原生支持传统意义上的方法重载？
11. 什么是多态？为什么多态可以减少 `if isinstance()`？
12. 鸭子类型与抽象基类有什么区别？
13. `ABC` 和 `Protocol` 如何选择？
14. `super()` 的作用是什么？为什么不建议到处直接调用父类名称？
15. `__new__()` 和 `__init__()` 的执行顺序是什么？
16. `__str__()` 和 `__repr__()` 的使用场景有什么不同？
17. `__eq__()` 和 `__hash__()` 为什么需要一起考虑？
18. 为什么对象作为字典 Key 时必须满足什么条件？
19. 上下文管理器解决了什么问题？
20. 描述符和 `property` 有什么关系？
21. Mixin 应该具有什么特点？
22. 什么是 MRO？如何查看类的 MRO？
23. 什么时候使用 `dataclass`，什么时候手写普通类？
24. `__slots__` 有什么优点和限制？
25. 为什么一个类不应该承担太多职责？
26. 什么是单一职责原则？在本题的 `TaskService` 中如何体现？
27. 什么是开闭原则？如何让任务系统支持新增任务类型？
28. 什么是依赖倒置？为什么 `TaskService` 不应该直接依赖 `MockModelClient`？
29. 如果一个类需要很多构造参数，可能说明了什么问题？
30. 如何判断一个继承关系是否违反了里氏替换原则？

## 16. 常见错误要求

完成练习时必须主动避免以下写法：

### 16.1 把所有数据都写成公开属性

错误示例：

```python
account.balance = -100
```

如果属性有业务不变量，应通过方法或属性 setter 控制。

### 16.2 滥用继承

不要因为两个类都有相同字段，就直接让一个类继承另一个类。先判断它们是否真正存在稳定的 is-a 关系。

### 16.3 父类方法复制到子类

如果子类只需要增加少量行为，应考虑调用 `super()`，不要复制整段父类代码。

### 16.4 巨大的 `if/elif` 类型判断

如果每增加一种类型都要修改核心流程，应考虑多态、注册表、策略模式或任务处理器。

### 16.5 类属性保存实例状态

错误示例：

```python
class User:
    permissions = []
```

这样所有实例可能共享同一个列表。实例独有的可变状态应在 `__init__()` 中创建。

### 16.6 过度使用魔术方法

魔术方法应让对象表现得符合 Python 直觉，而不是为了展示知识点堆叠几十个特殊方法。

### 16.7 过度封装

不是所有字段都需要复杂 getter/setter。只有存在校验、派生值、权限或状态约束时，封装才有明确价值。

## 17. 最终验收标准

### 17.1 基础能力

- [ ] 能独立定义类、构造对象并管理对象状态。
- [ ] 能正确使用实例方法、类方法和静态方法。
- [ ] 能区分实例属性、类属性和局部变量。
- [ ] 能使用 `property` 完成校验和只读属性。

### 17.2 面向对象核心能力

- [ ] 能使用封装保护对象不变量。
- [ ] 能使用继承复用公共行为。
- [ ] 能使用多态替代大量类型判断。
- [ ] 能使用抽象类定义稳定接口。
- [ ] 能根据场景选择继承、组合或依赖。

### 17.3 Python 高级能力

- [ ] 能实现常见魔术方法。
- [ ] 能实现上下文管理器和可调用对象。
- [ ] 能正确使用 `dataclass`、`__slots__` 和不可变对象。
- [ ] 能理解描述符和 MRO。
- [ ] 能使用 `Protocol` 和简单泛型。

### 17.4 工程能力

- [ ] 能将综合项目拆分为领域对象、接口、实现和服务。
- [ ] 能替换模型客户端、仓储和任务处理器。
- [ ] 能为异常、边界条件和状态变化编写测试。
- [ ] 能解释自己的类关系，而不是只展示代码能运行。
- [ ] 能发现并重构职责过多、继承过深和重复代码问题。

## 18. 推荐完成顺序

建议按照以下顺序完成，不要直接跳到综合项目：

1. 学生类：类、对象、属性、方法。
2. 银行账户：封装、异常、`property`。
3. 员工体系：继承、重写、`super()`。
4. 支付和导出器：多态、鸭子类型。
5. 存储接口：抽象类和依赖倒置。
6. 订单系统：组合、策略和职责拆分。
7. 购物车：魔术方法和上下文管理器。
8. 配置与金额对象：`dataclass`、不可变对象和 `__slots__`。
9. 描述符与插件注册：高级属性控制和扩展机制。
10. Mixin、`Protocol` 和泛型：高级类型与多继承。
11. 大模型应用任务系统：综合项目。

每完成一个阶段，先通过该阶段验收，再继续下一阶段。重点不是记住所有语法，而是能够根据业务需求设计出职责清晰、可扩展、可测试的对象模型。
