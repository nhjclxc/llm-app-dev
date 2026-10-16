import dataclasses
import sqlite3
import time
from datetime import datetime


"""
pip install sqlite3
uv add sqlite3

序号	API & 描述
1	sqlite3.connect(database [,timeout ,other optional arguments])
该 API 打开一个到 SQLite 数据库文件 database 的链接。您可以使用 ":memory:" 来在 RAM 中打开一个到 database 的数据库连接，而不是在磁盘上打开。如果数据库成功打开，则返回一个连接对象。

当一个数据库被多个连接访问，且其中一个修改了数据库，此时 SQLite 数据库被锁定，直到事务提交。timeout 参数表示连接等待锁定的持续时间，直到发生异常断开连接。timeout 参数默认是 5.0（5 秒）。

如果给定的数据库名称 filename 不存在，则该调用将创建一个数据库。如果您不想在当前目录中创建数据库，那么您可以指定带有路径的文件名，这样您就能在任意地方创建数据库。

2	connection.cursor([cursorClass])
该例程创建一个 cursor，将在 Python 数据库编程中用到。该方法接受一个单一的可选的参数 cursorClass。如果提供了该参数，则它必须是一个扩展自 sqlite3.Cursor 的自定义的 cursor 类。

3	cursor.execute(sql [, optional parameters])
该例程执行一个 SQL 语句。该 SQL 语句可以被参数化（即使用占位符代替 SQL 文本）。sqlite3 模块支持两种类型的占位符：问号和命名占位符（命名样式）。

例如：cursor.execute("insert into people values (?, ?)", (who, age))

4	connection.execute(sql [, optional parameters])
该例程是上面执行的由光标（cursor）对象提供的方法的快捷方式，它通过调用光标（cursor）方法创建了一个中间的光标对象，然后通过给定的参数调用光标的 execute 方法。

5	cursor.executemany(sql, seq_of_parameters)
该例程对 seq_of_parameters 中的所有参数或映射执行一个 SQL 命令。

6	connection.executemany(sql[, parameters])
该例程是一个由调用光标（cursor）方法创建的中间的光标对象的快捷方式，然后通过给定的参数调用光标的 executemany 方法。

7	cursor.executescript(sql_script)
该例程一旦接收到脚本，会执行多个 SQL 语句。它首先执行 COMMIT 语句，然后执行作为参数传入的 SQL 脚本。所有的 SQL 语句应该用分号 ; 分隔。

8	connection.executescript(sql_script)
该例程是一个由调用光标（cursor）方法创建的中间的光标对象的快捷方式，然后通过给定的参数调用光标的 executescript 方法。

9	connection.total_changes()
该例程返回自数据库连接打开以来被修改、插入或删除的数据库总行数。

10	connection.commit()
该方法提交当前的事务。如果您未调用该方法，那么自您上一次调用 commit() 以来所做的任何动作对其他数据库连接来说是不可见的。

11	connection.rollback()
该方法回滚自上一次调用 commit() 以来对数据库所做的更改。

12	connection.close()
该方法关闭数据库连接。请注意，这不会自动调用 commit()。如果您之前未调用 commit() 方法，就直接关闭数据库连接，您所做的所有更改将全部丢失！

13	cursor.fetchone()
该方法获取查询结果集中的下一行，返回一个单一的序列，当没有更多可用的数据时，则返回 None。

14	cursor.fetchmany([size=cursor.arraysize])
该方法获取查询结果集中的下一行组，返回一个列表。当没有更多的可用的行时，则返回一个空的列表。该方法尝试获取由 size 参数指定的尽可能多的行。

15	cursor.fetchall()
该例程获取查询结果集中所有（剩余）的行，返回一个列表。当没有可用的行时，则返回一个空的列表。
"""

def open_db(filename: str = 'test.db') -> sqlite3.Connection:
    conn = sqlite3.connect(filename)
    return conn

def close_db(conn: sqlite3.Connection) -> None:
    conn.close()
    pass

def create_table(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS COMPANY
                      (
                     ID      INT PRIMARY KEY NOT NULL,
                     NAME    TEXT            NOT NULL,
                     AGE     INT             NOT NULL,
                     ADDRESS CHAR(50),
                     SALARY  REAL
                      );''')
    # 每次进行写操作都要提交
    conn.commit()
    print("数据表创建成功")
    pass

@dataclasses.dataclass
class Company:
    def __init__(self, id: int, name: str, age: int, address: str, salary: float) -> None:
        self.id = id
        self.name = name
        self.age = age
        self.address = address
        self.salary = salary


def insert_data(conn: sqlite3.Connection, datas: list[Company]) -> int:
    cursor = conn.cursor()
    counter = 0
    for data in datas:

        id = data.id
        if id is None or id <= 0:
            id = counter + int(time.time())

        # 使用 '?' 进行参数占位，防止sql注入
        cursor.execute("INSERT INTO COMPANY (ID, NAME, AGE, ADDRESS, SALARY) VALUES (?, ?, ?, ?, ?)",
                       (id, data.name, data.age, data.address, data.salary))
        counter += 1
    conn.commit()
    return counter

def gen_data() -> list[Company]:
    datas = []
    # datas.append({"ID": 1, "NAME": 'Paul', "AGE": 32, "ADDRESS": 'California', "SALARY": 20000.00})
    # datas.append({"ID": 2, "NAME": 'Allen', "AGE": 25, "ADDRESS": 'Texas', "SALARY": 15000.00})
    # datas.append({"ID": 3, "NAME": 'Teddy', "AGE": 23, "ADDRESS": 'Norway', "SALARY": 20000.00})
    # datas.append({"ID": 4, "NAME": 'Mark', "AGE": 25, "ADDRESS": 'Rich-Mond', "SALARY": 65000.00})
    datas.append(Company(0, 'Paul',32,'California',20000.00))
    datas.append(Company(0, 'Allen',25,'Texas',15000.00))
    datas.append(Company(0, 'Teddy',23,'Norway',20000.00))
    datas.append(Company(0, 'Mark',25,'Rich-Mond',65000.00))
    return datas


def select_data(conn: sqlite3.Connection) -> list[Company]:
    cursor = conn.cursor()
    cur = cursor.execute("SELECT id, name, age, address, salary  from COMPANY")
    datas = []
    for row in cur:
        datas.append(Company(*row))
    return datas

def update_data(conn: sqlite3.Connection, data: Company) -> None:
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE COMPANY SET NAME = ?, ADDRESS = ? WHERE ID = ?",
        (data.name, data.address, data.id),
    )
    conn.commit()
    print("修改数据成功")
    pass

def delete_data(conn: sqlite3.Connection, id) -> None:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM COMPANY WHERE ID = ?", (id,))
    conn.commit()

    print("删除数据成功")
    pass


def transaction_test(conn):

    # commit_flag = False
    # try:
    #     # ...
    #     conn.commit()
    #     commit_flag = True
    # except Exception as e:
    #     print(e)
    #     conn.rollback()
    # finally:
    #     if not commit_flag:
    #         conn.rollback()

    # 使用 with 来管理整个连接，包括commit提交事务和rollback回滚
    with conn:
        data1 = Company(666, 'Mark666', 25, 'Rich-Mond666', 65000.00)
        data2 = Company(667, 'Mark667', 25, 'Rich-Mond667', 65000.00)
        for data in [data1, data2]:
            cursor.execute("INSERT INTO COMPANY (ID, NAME, AGE, ADDRESS, SALARY) VALUES (?, ?, ?, ?, ?)",
                           (data.id, data.name, data.age, data.address, data.salary))
        print("事务插入成功")
        cursor.execute("DELETE FROM COMPANY WHERE ID = ?", (data1.id,))
        print("事务删除成功")
        # 模拟异常
        x = 1 / 0
        print("事务完成")
        pass

    pass


if __name__ == '__main__':

    print()

    conn = open_db('test.db')
    print ("数据库打开成功")
    cursor = conn.cursor()

    create_table(conn)

    datas = gen_data()
    insert_data(conn, datas)

    select_datas = select_data(conn)
    print(select_datas)

    data0 = datas[0]
    data0.name = "zhangsan"
    data0.address = "beijing"
    update_data(conn, data0)
    select_datas = select_data(conn)
    print(select_datas)

    delete_data(conn, 4)


    # 模拟事务
    transaction_test(conn)


    # 关闭数据库
    close_db(conn)
    pass



