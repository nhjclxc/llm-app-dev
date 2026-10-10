

import dataclasses

# https://docs.python.org/zh-cn/3/library/dataclasses.html
# python 的 dataclasses 库必会的方法

# 快速定义“只负责存数据”的类。

from dataclasses import dataclass, field, asdict

@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""
    name: str
    unit_price: float
    quantity_on_hand: int = 0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        # __post_init__ 在 __init__ 函数之后被调用，一般用于确定参数是否合法
        if self.unit_price < 2:
            raise ValueError("age 不能小于 2")

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand

ii = InventoryItem("Tomato", 11)
print(ii.name, ii.unit_price, ii.quantity_on_hand)
print(ii.total_cost())
print(InventoryItem("Tomato", 2.5, 10).total_cost())

print(asdict(ii))
