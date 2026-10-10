
-- 教程 https://www.runoob.com/postgresql/postgresql-tutorial.html

-- PostgreSQL 语法： https://www.runoob.com/postgresql/postgresql-syntax.html

-- 各种常用数据类型
-- 名字	存储长度	描述	范围
-- smallint	2 字节	小范围整数	-32768 到 +32767
-- integer	4 字节	常用的整数	-2147483648 到 +2147483647
-- bigint	8 字节	大范围整数	-9223372036854775808 到 +9223372036854775807
-- decimal	可变长	用户指定的精度，精确	小数点前 131072 位；小数点后 16383 位
-- numeric	可变长	用户指定的精度，精确	小数点前 131072 位；小数点后 16383 位
-- real	4 字节	可变精度，不精确	6 位十进制数字精度
-- double precision	8 字节	可变精度，不精确	15 位十进制数字精度
-- smallserial	2 字节	自增的小范围整数	1 到 32767
-- serial	4 字节	自增整数	1 到 2147483647
-- bigserial	8 字节	自增的大范围整数	1 到 9223372036854775807

CREATE TABLE postgres_all_types (
    id BIGSERIAL PRIMARY KEY,

    -- ==========================================
    -- 1. 整数类型
    -- ==========================================

    col_smallint       SMALLINT Not Null ,
    col_integer        INTEGER,
    col_bigint         BIGINT,
    -- SERIAL 可以自增   ，INT 类型不能自增
    col_smallserial    SMALLSERIAL,
    col_serial         SERIAL,
    col_bigserial      BIGSERIAL,


    -- ==========================================
    -- 2. 精确数值
    -- ==========================================

    col_numeric        NUMERIC,
    col_numeric_10_2   NUMERIC(10, 2),
    col_decimal        DECIMAL(10, 2),


    -- ==========================================
    -- 3. 浮点数
    -- ==========================================

    col_real           REAL,
    col_double         DOUBLE PRECISION,


    -- ==========================================
    -- 4. 布尔
    -- ==========================================

    col_boolean        BOOLEAN,


    -- ==========================================
    -- 5. 字符串
    -- ==========================================

    col_char           CHAR(10),
    col_varchar        VARCHAR(255),
    col_text           TEXT,


    -- ==========================================
    -- 7. 日期时间
    -- ==========================================

    col_date            DATE,

    col_time            TIME,

    col_time_tz         TIME WITH TIME ZONE,

    col_timestamp       TIMESTAMP,

    col_timestamp_tz    TIMESTAMP WITH TIME ZONE,

    col_interval        INTERVAL,


    -- ==========================================
    -- 10. JSON
    -- ==========================================

    col_json              JSON,

    col_jsonb             JSONB,

    col_jsonpath          JSONPATH,


    -- ==========================================
    -- 22. 数组
    -- ==========================================

    col_int_array              INTEGER[],

    col_text_array             TEXT[],

    col_2d_int_array           INTEGER[][]

);


insert into postgres_all_types
    (id, col_smallint, col_integer, col_bigint, col_smallserial, col_serial, col_bigserial,
     col_numeric, col_numeric_10_2, col_decimal, col_real, col_double,
     col_boolean, col_char, col_varchar, col_text,
     col_date, col_time, col_time_tz, col_timestamp, col_timestamp_tz, col_interval,
     col_json, col_jsonb, col_jsonpath, col_int_array, col_text_array, col_2d_int_array)
VALUES (1,2,3,4,5,6, 9,
        12.123, 23.456, 56.789, 12.123, 23.345,
        true, 'c', 'ascas', '自增按时',
        '2026-09-28', '15:19:29', '15:19:29+08','2026-09-28 12:30:00','2026-09-28 12:30:00+08','1 day 2 hours',
        '{"name":"Tom","age":18}','{"name":"Tom","age":18}','$.name',
        ARRAY[1,2,3], ARRAY['a','b','c'], ARRAY[[1,2],[3,4]]
        );


-- 创建一个 schema 模式，可以理解为是mysql里面的一个试图
CREATE SCHEMA IF NOT EXISTS myschema;

CREATE TABLE myschema.postgres_all_types (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- 整数
    small_int       SMALLINT,
    int_value       INTEGER,
    big_int         BIGINT,

    -- 小数
    decimal_value   NUMERIC(10, 2),
    real_value      REAL,
    double_value    DOUBLE PRECISION,

    -- 字符串
    char_value      CHAR(10),
    varchar_value   VARCHAR(255),
    text_value      TEXT,

    -- 布尔
    is_active       BOOLEAN,

    -- 日期时间
    birthday        DATE,
    start_time      TIME,
    created_at      TIMESTAMP,
    updated_at      TIMESTAMPTZ,
    duration        INTERVAL,

    -- JSON
    json_value      JSON,
    jsonb_value     JSONB,

    -- 数组
    int_array       INTEGER[],
    text_array      TEXT[]
);

-- 单次数据同步
insert into myschema.postgres_all_types (small_int)
select col_smallint
from postgres_all_types;

-- 实时数据同步，使用 Trigger 触发器


-- 通过 RETURNING 获取sql执行成功后对应列的值
INSERT INTO postgres_all_types(col_varchar)
VALUES ('Tom')
RETURNING id;

UPDATE postgres_all_types
SET col_varchar = 'Jerry'
WHERE id = 3
RETURNING id, col_varchar;
-- RETURNING *;

-- 设置自增id
CREATE TABLE users (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name varchar(32)
);
insert into users(name) values ('zs1'), ('zs2'), ('zs3'), ('zs4'), ('zs5');
insert into users(name) values ('zs6'), ('zs7'), ('zs8'), ('zs9'), ('zs10');


-- 字符串拼接，使用 || 进行拼接
select 'aaa' || ' -> ' || 'bbb';
select id || ' -> ' || name from users;


-- 分页特性
-- 通用的分页特性
select *
from users
ORDER BY id
limit 3 offset 3*(1-1);
-- limit 3 offset 3*(2-1);
-- limit 3 offset 3*(3-1);


-- postgresql独有的分页特性
-- FETCH FIRST 10 ROWS ONLY; 要搭配OFFSET使用
select *
from users
ORDER BY id
OFFSET 3*(1-1)
FETCH FIRST 3 ROWS ONLY;



CREATE TABLE COMPANY(
   ID INT PRIMARY KEY     NOT NULL,
   NAME           TEXT    NOT NULL,
   AGE            INT     NOT NULL,
   ADDRESS        CHAR(50),
   SALARY         REAL,
   JOIN_DATE      DATE
);
INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY,JOIN_DATE) VALUES (1, 'Paul', 32, 'California', 20000.00,'2001-07-13');
INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,JOIN_DATE) VALUES (2, 'Allen', 25, 'Texas', '2007-12-13');
INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY,JOIN_DATE) VALUES (3, 'Teddy', 23, 'Norway', 20000.00, DEFAULT );
INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY,JOIN_DATE) VALUES (4, 'Mark', 25, 'Rich-Mond ', 65000.00, '2007-12-13' ), (5, 'David', 27, 'Texas', 85000.00, '2007-12-13');

-- https://static.jyshare.com/download/company.sql
INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)
VALUES (11, 'Paul', 32, 'California', 20000.00 ),
(12, 'Allen',25, 'Texas', 15000.00 ),
(13, 'Teddy',23, 'Norway', 20000.00 ),
(14, 'Mark', 25, 'Rich-Mond ', 65000.00 ),
(15, 'David',27, 'Texas', 85000.00 ),
(16, 'Kim',  22, 'South-Hall', 45000.00 ),
(17, 'James',24, 'Houston', 10000.00 );


-- PostgreSQL 运算符
-- 运算符	描述	实例
-- +	加	a + b 结果为 5
-- -	减	a - b 结果为 -1
-- *	乘	a * b 结果为 6
-- /	除	b / a 结果为 1
-- %	模（取余）	b % a 结果为 1
-- ^	指数	a ^ b 结果为 8
-- |/	平方根	|/ 25.0 结果为 5
-- ||/	立方根	||/ 27.0 结果为 3
-- !	阶乘	5 ! 结果为 120
-- !!	阶乘（前缀操作符）	!! 5 结果为 120


-- PostgreSQL 表达式
-- SELECT column1, column2, columnN
-- FROM table_name
-- WHERE [CONDITION | EXPRESSION];

-- 布尔表达式
select * from company where SALARY = 10000;
-- 数字表达式
SELECT (17 + 6) AS ADDITION ;
select count(*) from company;
-- 日期表达式
select current_time, current_timestamp;


-- PostgreSQL WHERE 子句
-- PostgreSQL AND & OR 运算符

-- PostgreSQL LIKE 子句
-- :: 是 PostgreSQL 的类型转化语法
SELECT * FROM COMPANY WHERE AGE::text LIKE '2%';


-- PostgreSQL LIMIT 子句
-- LIMIT [no of rows] OFFSET [row num]
select *
from users
LIMIT 3 OFFSET 3*(2-1)
;

-- PostgreSQL ORDER BY 语句
-- ORDER BY 用于对一列或者多列数据进行升序（ASC）或者降序（DESC）排列。

select *
from users
-- ORDER BY id
-- ORDER BY id asc
ORDER BY id desc
;

-- PostgreSQL GROUP BY 语句
-- 在 PostgreSQL 中，GROUP BY 语句和 SELECT 语句一起使用，用来对相同的数据进行分组。
-- GROUP BY 在一个 SELECT 语句中，放在 WHERE 子句的后面，ORDER BY 子句的前面。
select name, sum(SALARY)
from company
group by name;

-- PostgreSQL WITH 子句
-- WITH 子句有助于将复杂的大型查询分解为更简单的表单，便于阅读。这些语句通常称为通用表表达式（Common Table Express， CTE），也可以当做一个为查询而存在的临时表。
-- WITH 子句是在多次执行子查询时特别有用，允许我们在查询中通过它的名称(可能是多次)引用它。
-- WITH 子句在使用前必须先定义。
-- WITH
--    name_for_summary_data AS (
--       SELECT Statement)
--    SELECT columns
--    FROM name_for_summary_data
--    WHERE conditions <=> (
--       SELECT column
--       FROM name_for_summary_data)
--    [ORDER BY columns]
-- WITH 递归
-- 在 WITH 子句中可以使用自身输出的数据。
-- 公用表表达式 (CTE) 具有一个重要的优点，那就是能够引用其自身，从而创建递归 CTE。递归 CTE 是一个重复执行初始 CTE 以返回数据子集直到获取完整结果集的公用表表达式。
create table tab_origin(
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pid BIGINT NOT NULL ,
    value varchar(32) NOT NULL
);
insert into tab_origin(pid, value) values (0, '1'), (0, '2'), (0, '3');
insert into tab_origin(pid, value) values (1, '1-1'), (1, '1-2');
insert into tab_origin(pid, value) values (4, '1-1-1'), (4, '1-1-2');
insert into tab_origin(pid, value) values (2, '2-1'), (5, '2-1-1');
insert into tab_origin(pid, value) values (3, '3-1');

-- 【父找子】查询id=1下的所有数据
-- 通过内部的第一个sql查询出id=1的节点信息，此时id=1的节点信息已在with_tab_origin
-- 之后再通过 union all 去聚合 tab_origin 和 with_tab_origin 的关联查询结果
WITH RECURSIVE with_tab_origin AS (
    -- 第一层：找到根节点
    -- ① 锚点查询：找到起点
    SELECT id, pid, value
    FROM tab_origin
    WHERE id = 1

    UNION ALL

    -- 后续层：不断寻找当前节点的子节点
    -- ② 递归查询：寻找子节点
    SELECT t.id, t.pid, t.value
    FROM tab_origin t
    JOIN with_tab_origin w
        ON t.pid = w.id
)
SELECT id, pid, value
FROM with_tab_origin
ORDER BY id;

-- 【子找父】
with recursive with_tab_origin2 as (
    select id, pid, value
    from tab_origin
    where id = 9

    union all

    select o.id, o.pid,o.value
    from tab_origin o
    join with_tab_origin2 o2 on o.id = o2.pid
)
select *
from with_tab_origin2
order by id;

-- PostgreSQL HAVING 子句
-- HAVING 子句可以让我们筛选分组后的各组数据。
-- WHERE 子句在所选列上设置条件，而 HAVING 子句则在由 GROUP BY 子句创建的分组上设置条件。
select name, count(name)
from company
group by name
having count(name) > 1;


-- PostgreSQL DISTINCT 关键字
-- 在 PostgreSQL 中，DISTINCT 关键字与 SELECT 语句一起使用，用于去除重复记录，只获取唯一的记录。
-- 我们平时在操作数据时，有可能出现一种情况，在一个表中有多个重复的记录，当提取这样的记录时，DISTINCT 关键字就显得特别有意义，它只获取唯一一次记录，而不是获取重复记录。
INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)
VALUES (8, 'Paul', 32, 'California', 20000.00 ),(9, 'Allen', 25, 'Texas', 15000.00 );

select name from company order by name;
select distinct name from company order by name;



-- :: 是 PostgreSQL 的类型转化语法
-- -- integer → text
-- age::text
--
-- -- text → integer
-- '123'::integer
--
-- -- text → date
-- '2026-09-29'::date
--
-- -- timestamp → date
-- created_at::date
--
-- -- integer → numeric
-- age::numeric
--
-- -- json → jsonb
-- data::jsonb

-- postgresql 的json和jsonb的区别
-- | 对比      | `json`      | `jsonb`          |
-- | ------- | ----------- | ---------------- |
-- | 存储方式    | 保存 JSON 文本  | 保存解析后的二进制结构      |
-- | 写入速度    | 较快          | 稍慢，需要解析          |
-- | 查询速度    | 较慢          | **较快**           |
-- | 索引      | 支持有限        | **支持 GIN 等索引**   |
-- | 保留空格    | ✅           | ❌                |
-- | 保留键的顺序  | ✅           | ❌                |
-- | 重复 key  | 保留          | **后面的覆盖前面的**     |
-- | JSON 解析 | 查询时处理       | 写入时处理            |
-- | 推荐场景    | 单纯保存原始 JSON | **需要查询/索引 JSON** |

-- jsonb会重新对json对key进行排序，
SELECT '{"name":"Tom","age":18}'::json;
SELECT '{"name":"Tom","age":18}'::jsonb;

-- jsonb重复key会被后面的覆盖
SELECT '{"name":"Tom","name":"Jerry"}'::json;
SELECT '{"name":"Tom","name":"Jerry"}'::jsonb;

create table tab_json (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- JSON
    json_value      JSON,
    jsonb_value     JSONB
);


insert into tab_json(json_value, jsonb_value)
values
('{"domain_id":"124","domain_name":"demo01.com","provider_name":"alyun","provider_sub_account":"cdn支撑","service_area":"mainland","change_history":[{"effective_end_time":1790319600,"effective_start_time":1786464000,"flow_isp":null,"flow_template":"demo01.com","sampling_ratio":1}]}', '{"domain_id":"124","domain_name":"demo01.com","provider_name":"alyun","provider_sub_account":"cdn支撑","service_area":"mainland","change_history":[{"effective_end_time":1790319600,"effective_start_time":1786464000,"flow_isp":null,"flow_template":"demo01.com","sampling_ratio":1}]}')
,('{"domain_id":"125","domain_name":"demo02.com","provider_name":"alyun","provider_sub_account":"cdn支撑","service_area":"mainland","change_history":[{"effective_end_time":1791270000,"effective_start_time":1787414400,"flow_isp":null,"flow_template":"demo02.com","sampling_ratio":1}]}', '{"domain_id":"125","domain_name":"demo02.com","provider_name":"alyun","provider_sub_account":"cdn支撑","service_area":"mainland","change_history":[{"effective_end_time":1791270000,"effective_start_time":1787414400,"flow_isp":null,"flow_template":"demo02.com","sampling_ratio":1}]}')
,('{"domain_id":"126","domain_name":"demo03.com","provider_name":"alyun","provider_sub_account":"cdn支撑","service_area":"mainland","change_history":[{"effective_end_time":1792220400,"effective_start_time":1788364800,"flow_isp":null,"flow_template":"demo03.com","sampling_ratio":1}]}', '{"domain_id":"126","domain_name":"demo03.com","provider_name":"alyun","provider_sub_account":"cdn支撑","service_area":"mainland","change_history":[{"effective_end_time":1792220400,"effective_start_time":1788364800,"flow_isp":null,"flow_template":"demo03.com","sampling_ratio":1}]}')

-- jsonb的查询特性
-- 其中的 '->>'表示 从 JSON 中取出 name，并且结果转换成 text。
select *
from tab_json
where json_value->>'domain_name' = 'demo02.com';
-- where jsonb_value->>'domain_name' = 'demo02.com';

-- 7. jsonb 最大的优势之一：索引
CREATE INDEX tab_json_jsonb_value
ON tab_json
USING GIN (jsonb_value);

-- 当jsonb创建索引之后可以使用jsonb的索引进行查询
-- '@>' 表示左边的json是否包含右边的json
SELECT *
FROM tab_json
WHERE jsonb_value @> '{"domain_name":"demo02.com"}';


-- json 适合保存 JSON 原文
-- jsonb  -> 适合 JSON 数据查询 -> 适合建立索引 -> 适合复杂 JSON 条件查询



-- PostgreSQL 约束
-- NOT NULL 约束
-- UNIQUE 约束
-- PRIMARY KEY
CREATE TABLE COMPANY1(
   ID INT PRIMARY KEY     NOT NULL,
   NAME           TEXT    NOT NULL,
   AGE            INT     NOT NULL UNIQUE,
   ADDRESS        CHAR(50),
   SALARY         REAL
);





