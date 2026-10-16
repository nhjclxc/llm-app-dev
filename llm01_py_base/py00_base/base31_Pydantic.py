
import pydantic
from pydantic import BaseModel, Field, field_validator, model_validator


# python 的 Pydantic 库必会的方法



# 1. BaseModel

class Order(BaseModel):
    name: str = Field(min_length=1, max_length=20)
    order_id: str

# 使用 BaseModel 定义的类，具有自动校验 + 类型转换 + 序列化的功能
class User(BaseModel):
    # 7. Field() 。
    name: str = Field(min_length=1, max_length=20)
    age: int = Field(default = 18, ge=0, le=150)
#     Field(default=...,min_length=1, max_length=100,gt=0,ge=0,lt=100,le=100,description="用户年龄")
    # 设置默认值 None
    desc: str | None = None
    desc2: str | None = "description"
    # 9. list / dict / 嵌套模型
    items: list[Order] = Field(default_factory=list)


    pass

class Admin(BaseModel):

    # 10. @ field_validator 自定义字段校验。
    password: str
    @field_validator("password")
    @classmethod
    def validate_password(cls, value):
        if value is None:
            # 未设置密码，则系统随机生成一个密码
            return value
        if value.strip() == "":
            return ""
        if len(value) < 8 or len(value) > 16:
            raise pydantic.ValidationError("Password must be at least 8 or max 16 characters long")
        print("密码校验通过")

        return value

    # 11. @ model_validator 对整个 Model 进行校验。
    password_confirm: str
    @model_validator(mode="after")
    def validate_password(self):

        if self.password != self.password_confirm:
            raise pydantic.ValidationError("两次输入的密码不一致")

        print("校验通过")

        pass

    pass


if __name__ == "__main__":
    user = User(name="tom", age=11)
    print(user)

    # 2. model_validate() 将字典转化为对象  dict → Pydantic Model
    # 实际开发中一般用于将接口数据转化为py对象
    data = {"name": "tom", "age": 12}
    u2 = User.model_validate(data)
    print(u2, type(u2))

    # pydantic_core._pydantic_core.ValidationError: 1 validation error for User age
    # data3 = {"name": "tom", "age1": 12}
    # u3 = User.model_validate(data3)
    # print(u3, type(u3))

    #3. model_dump() 把 Pydantic Model 转成 Python dict
    u2_dict = u2.dict()
    u2_dict2 = u2.model_dump()
    print(u2_dict, type(u2_dict))
    print(u2_dict2, type(u2_dict2))

    # 4. model_dump_json() 把 Pydantic Model 转成 JSON 字符串。
    u2_json = u2.model_dump_json()
    print(u2_json, type(u2_json))

    # 5. model_validate_json()  把 JSON 字符串直接转换成 Pydantic Model。
    u3 = User.model_validate_json(u2_json)
    print(u3, type(u3))

    u3.items.append(Order(name="joe", order_id="123"))
    u3.items.append(Order(name="joe2", order_id="123"))
    print(u3, type(u3))

    # raise pydantic.ValidationError("Password must be at least 8 or max 16 characters long")
    # admin = Admin(password="passw")

    # raise pydantic.ValidationError("两次输入的密码不一致")
    # Admin = Admin(password="<PASSWORD>", password_confirm="<PASSWORD>1")

    admin = Admin(password="<PASSWORD>", password_confirm="<PASSWORD>")
    print("admin: ", admin)

    pass


