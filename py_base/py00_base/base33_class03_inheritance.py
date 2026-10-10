import random


## 5. 阶段 3：继承与方法重写
### 5.1 练习七：员工体系

class Person:
    def __init__(self, name, age = 18):
        self.name = name
        self.age = age
    def work(self):
        print("person is working")

class Employee(Person):

    def __init__(self, employee_id: int, name: str, salary: float):
        super().__init__(name)
        self.employee_id = employee_id
        self.name = name
        self.salary = salary

    def calculate_salary(self):
        """ 计算工资 """
        print("计算工资")

    def work(self):
        """ 描述工作 """
        print("描述工作")

    def summary(self):
        """ 员工摘要 """
        print("员工摘要")

    def __str__(self):
        """ 员工信息 """
        print(f"Employee(employee_id={self.employee_id}, name={self.name}, salary={self.salary})")



class FullTimeEmployee(Employee):
    """ 固定工资 + 津贴； """

    def __init__(self, employee_id: int, name: str, salary: float):
        super().__init__(employee_id, name, salary)

    def calculate_salary(self):
        r = random.randint(1000, 5000)
        return self.salary + r

    def work(self):
        print("固定工资 + 津贴")


class PartTimeEmployee(Employee):
    """ 小时工资 × 工作小时； """

    def __init__(self, employee_id: int, name: str, salary: float):
        super().__init__(employee_id, name, salary)

    def calculate_salary(self):
        r = random.randint(38, 50)
        return self.salary * r

    def work(self):
        print("小时工资 × 工作小时")


class CommissionEmployee(Employee):
    """ 基础工资 + 销售提成。 """
    def __init__(self, employee_id: int, name: str, salary: float):
        super().__init__(employee_id, name, salary)

    def calculate_salary(self):
        r = random.randint(5000, 8000)
        return self.salary + r

    def work(self):
        print("基础工资 + 销售提成")


# class Manager(Employee):
#     def __init__(self, name):
#         super().__init__(name)
#

class Payment:
    def __init__(self, account_id):
        self.account_id = account_id
        pass
    def pay(self, amount: float) -> str:
        return ""
    pass

class AlipayPayment(Payment):
    def __init__(self, account_id):
        super().__init__(account_id)
    def pay(self, amount: float) -> str:
        return ""
    pass

class WechatPayment(Payment):
    def __init__(self, account_id):
        super().__init__(account_id)
    def pay(self, amount: float) -> str:
        return ""
    pass

class BankCardPayment(Payment):
    def __init__(self, account_id):
        super().__init__(account_id)
    def pay(self, amount: float) -> str:
        return ""
    pass

class MockPayment(Payment):
    def __init__(self, account_id):
        super().__init__(account_id)
    def pay(self, amount: float) -> str:
        return ""
    pass


def checkout(payment_method, amount: float) -> str | None:
    # 利用python的鸭子类型
    payment_method.pay(amount)

    # match payment_method:
    #     case "AlipayPayment":
    #         return AlipayPayment("account_alipay").pay(amount)
    #     case "WechatPayment":
    #         return WechatPayment("account_wxpay").pay(amount)
    #     case "BankCardPayment":
    #         return BankCardPayment("account_bankpay").pay(amount)
    #     case "MockPayment":
    #         return MockPayment("account_mockpay").pay(amount)
    #     case _:
    #         raise NotImplementedError

class EmailSender:
    def send(self, message):
        print(f" EmailSender -> {message} ")

class SmsSender:
    def send(self, message):
        print(f" SmsSender -> {message} ")

class ConsoleSender:
    def send(self, message):
        print(f" ConsoleSender -> {message} ")

def notify(sender, message: str) -> None:
    sender.send(message)

# notify(EmailSender(), "email:msg")
# notify(SmsSender(), "sms:msg")
# notify(ConsoleSender(), "console:msg")




class ReportExporter:
    def export(self, rows: list[dict]) -> str:
        pass
    pass
class CsvExporter:
    def export(self, rows: list[dict]) -> str:
        pass
    pass
class JsonExporter:
    def export(self, rows: list[dict]) -> str:
        pass
    pass
class MarkdownExporter:
    def export(self, rows: list[dict]) -> str:
        pass
    pass

def export_report(exporter, rows: list[dict]) -> str:
    return exporter.export(rows)


from abc import ABC, abstractmethod


class Storage(ABC):
    @abstractmethod
    def save(self, key: str, value: str) -> None:
        pass

    @abstractmethod
    def get(self, key: str) -> str | None:
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

class MemoryStorage(Storage):

    def save(self, key: str, value: str) -> None:
        print(f" MemoryStorage -> {key} -> {value} ")
        pass

    def get(self, key: str) -> str | None:
        value = ""
        print(f" MemoryStorage -> {key} -> {value} ")
        pass

    def delete(self, key: str) -> bool:
        value = ""
        print(f" MemoryStorage -> {key} -> {value} ")
        return True