
-- ★★★★★ 第一层：必须掌握
--
-- 1. Database / Schema
-- 2. PostgreSQL 数据类型
-- 3. TIMESTAMP / TIMESTAMPTZ
-- 4. SERIAL / IDENTITY
-- 5. JSON / JSONB
-- 6. ARRAY
-- 7. UUID
-- 8. INET / CIDR
-- 9. NULL 行为
-- 10. PostgreSQL 类型转换
-- 类型转换1: '字段::目标类型'，如：123::text、'123'::integer
-- 类型转换2: 'cast(表达式 as 类型)'
SELECT 123::text;
SELECT ('123')*3;
SELECT ('123'::integer)*2;
SELECT CAST(123 AS text);
SELECT '550e8400-e29b-41d4-a716-446655440000'::uuid;
SELECT '{"name":"Tom"}'::jsonb;
SELECT '{1,2,3}'::integer[];
SELECT ARRAY[1,2,3]::text[];

-- 11. INSERT ... ON CONFLICT
--     数据不存在就 INSERT，数据已经存在就 UPDATE，或者直接忽略。
CREATE TABLE COMPANY(
   ID INT PRIMARY KEY     NOT NULL,
   NAME           TEXT    NOT NULL,
   AGE            INT     NOT NULL,
   ADDRESS        CHAR(50),
   SALARY         REAL,
   JOIN_DATE      DATE
);

INSERT INTO company (id, name, age)
VALUES (1, 'Tom2', 28)
ON CONFLICT (id)
DO UPDATE
SET
    name = EXCLUDED.name,
    age = EXCLUDED.age;


-- 12. RETURNING
-- 13. CTE / WITH
-- 14. Window Function
-- 15. LIMIT / OFFSET / FETCH
-- 16. PostgreSQL 函数
-- 17. EXPLAIN / EXPLAIN ANALYZE
-- 18. PostgreSQL 索引
-- 19. MVCC
-- 20. VACUUM





-- ★★★★☆ 第二层：数据库开发必须掌握
--
-- 21. Transaction
-- 22. READ COMMITTED
-- 23. REPEATABLE READ
-- 24. SERIALIZABLE
-- 25. Lock
-- 26. Deadlock
-- 27. Advisory Lock
-- 28. Foreign Key
-- 29. CHECK
-- 30. UNIQUE
-- 31. DEFERRABLE
-- 32. Partial Index
-- 33. Expression Index
-- 34. GIN
-- 35. GiST
-- 36. BRIN
-- 37. Partition
-- 38. Materialized View
-- 39. Trigger
-- 40. Function



-- ★★★☆☆ 第三层：数据库工程师必须掌握
--
-- 41. WAL
-- 42. Checkpoint
-- 43. Autovacuum
-- 44. Table Bloat
-- 45. pg_stat
-- 46. pg_catalog
-- 47. pg_hba.conf
-- 48. Backup
-- 49. PITR
-- 50. Streaming Replication
-- 51. Logical Replication
-- 52. Replication Slot
-- 53. Publication / Subscription
-- 54. Connection Pool
-- 55. PgBouncer


-- ★★☆☆☆ 第四层：高级 PostgreSQL
--
-- 56. Range / Multirange
-- 57. EXCLUDE Constraint
-- 58. RLS
-- 59. Custom Type
-- 60. Domain
-- 61. Operator
-- 62. Extension
-- 63. PostGIS
-- 64. Full Text Search
-- 65. GiST / SP-GiST
-- 66. BRIN
-- 67. SSI
-- 68. Predicate Lock
-- 69. Two-Phase Commit
-- 70. Foreign Data Wrapper



-- | 优先级 | 区别               | 一句话                                                                     |
-- | --- | ---------------- | ----------------------------------------------------------------------- |
-- | 1   | 架构               | PG 是整体式数据库，MySQL 有 Storage Engine 架构                                    |
-- | 2   | Schema           | PG 的 Schema 是独立命名空间，MySQL Database≈Schema                               |
-- | 3   | MVCC             | 两者 MVCC 实现完全不同                                                          |
-- | 4   | VACUUM           | PG 有 VACUUM，MySQL/InnoDB 主要依赖 Undo/Purge                                |
-- | 5   | 主键               | InnoDB 主键是聚簇索引，PG 主键默认不是                                                |
-- | 6   | 默认隔离级别           | PG=READ COMMITTED，InnoDB=REPEATABLE READ                                |
-- | 7   | Gap Lock         | InnoDB 有 Gap/Next-Key Lock，PG 没有同等机制                                    |
-- | 8   | JSON             | PG 有 JSON/JSONB，MySQL 有 JSON                                            |
-- | 9   | ARRAY            | PG 原生支持数组，MySQL 没有原生 ARRAY                                              |
-- | 10  | UUID             | PG 有原生 UUID 类型                                                          |
-- | 11  | INET/CIDR        | PG 原生支持 IP/网络类型                                                         |
-- | 12  | Range            | PG 原生支持范围类型                                                             |
-- | 13  | Index            | PG 有 B-tree/GIN/GiST/SP-GiST/BRIN 等丰富索引                                 |
-- | 14  | Partial Index    | PG 原生支持部分索引                                                             |
-- | 15  | Expression Index | PG 原生支持表达式索引                                                            |
-- | 16  | UPSERT           | PG=`ON CONFLICT`，MySQL=`ON DUPLICATE KEY`                               |
-- | 17  | RETURNING        | PG 对 DML 的 RETURNING 支持更完整                                              |
-- | 18  | 自增               | PG 常用 Identity/Sequence，MySQL 常用 AUTO_INCREMENT                         |
-- | 19  | 复制               | PG 核心是 WAL + Streaming/Logical Replication，MySQL 核心是 Binlog Replication |
-- | 20  | 扩展               | PG 的 Extension/自定义类型/运算符能力非常强                                           |






















