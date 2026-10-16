import torch

# print(torch.__version__)


def test_tensor1():
    # 2.1 从 Python 数据创建

    t1 = torch.tensor(1.23, dtype=torch.float64)
    print(t1,type(t1),t1.shape)

    t2 = torch.tensor([1,2,3])
    print(t2,type(t2),t2.shape)

    t3 = torch.tensor([[1,2,3], [4,5,6]])
    print(t3, type(t3), t3.shape)
    print(t3.dim()) # 纬度
    print(t3.size()) # 形状
    print(t3.numel()) # 元素个数
    print(t3.device)

    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    print(labels)
    print(labels.type())

    label2 = torch.tensor([True, False], dtype=torch.bool)
    print(label2, type(label2), label2.type())

    pass


def test_tensor2():
    #  2.2 常用工厂函数


    # 常用工厂函数
    # 全 0
    print(torch.zeros(1,2,3))
    # 全 1
    print(torch.ones(1,2,3))
    # 未初始化，读取前必须写入
    # `empty` 只分配内存，不会将元素清零，因此不要在填充前使用其中的值
    print(torch.empty(1,2,3))
    # 填充值 , 所有位置都用某个值填充
    print(torch.full((2,3), fill_value=666))
    # [0, 2, 4, 6, 8]
    print(torch.tensor([0,2,4,6,8]))
    print(torch.arange(0, 10, 2))
    # 等间隔，包含端点, steps 表示生成多少个数据
    print(torch.linspace(0,1, steps=5))
    # [0, 1) 均匀分布
    print(torch.rand(2,3))
    # 标准正态分布
    print(torch.randn(2,3))

    # 需要保持已有张量形式时，使用 *_like
    t1 = torch.ones(1, 2, 3)
    print(torch.zeros_like(t1).shape)
    print(torch.ones_like(t1).shape)


    pass


def test_tensor3():
    # 2.3 随机种子

    torch.manual_seed(21)
    a = torch.rand(3)
    print(a)

    torch.manual_seed(21)
    b = torch.rand(3)
    print(b)

    # 判断两个张量是否相同
    assert torch.equal(a, b)
    aa = torch.tensor([1,2,3])
    bb = torch.tensor([1,2,3])
    print(torch.equal(aa, bb))


    pass


def test_tensor4():
    # 3. 访问、切片与条件选择

    x = torch.arange(12).reshape(3,4)
    print(x)

    # 第 0 行第 1 列，标量张量
    print(x[0,1])
    # 第 1 行
    print(x[1])
    print(x[1, :])
    # 所有行的第 2 列
    print(x[:, 2])
    # 前两行、第 1 和第 2 列
    print(x[:2, 1:3])
    # 任意前置维度上的最后一列
    print(x[..., -1])
    print(x[:, -1])

    print("="*20)

    # 使用索引取张量会丢失维度，要保留维度要使用 切片或 `select` 后的 `unsqueeze`：
    print(x[:, 2], x[:, 2].shape) # 这个是取数据，会丢失一个维度
    print(x[1:2, :], x[1:2, :].shape) # 切片会保留维度
    print(x[:, 1:3], x[:, 1:3].shape)

    print("="*20)

    # 使用布尔掩码对元素进行筛选
    xx = torch.tensor([-2,-1,0,1,2])
    xx_masked = xx >= 0
    print(xx)
    print(xx_masked)
    # 使用掩码取原始数据
    print(xx[xx_masked])
    # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
    # print(xx[not xx_masked])
    # 对于tensor要想取反，必须使用 '~'
    print(xx[~xx_masked])
    # 注意，使用masked取出的数据都是一维张量，元素个数取决于masked里面True的数量


    pass

def test_tensor5():
    #  4. 张量运算
    # 4.1 逐元素运算

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])

    print(a + b)
    print(a - b)
    print(a * b)
    print(b / a)
    print(a ** 2)
    print(b ** a)

    print(torch.sqrt(b))
    print(torch.exp(b))
    print(torch.log(b))
    print(torch.abs(torch.tensor([-1.0, 2.0])))

    # 比较运算返回布尔张量：`a > 0`、`a == b`、`a != b`。逻辑组合可使用 `&`、`|`、`~`，条件选择可使用 `torch.where`：
    print(a > 0)
    print(a > 2)
    print(a == b)
    print(a != b)

    x = torch.tensor([1.0, -2.0, 3.0])
    # 第一个参数是条件
    # 第二个参数是输入张量，即要进行判断的
    # 第三个参数是 条件为 False 时，选择的值，这个张量的形状要与第二个参数的形状相同
    print(torch.where(x > 0, x, torch.zeros_like(x)))
    print(torch.where(x > 0, x, torch.tensor([11,22,33])))
    ...


def test_tensor6():
    # 4.2 广播机制

    # 当两个张量形状不同，PyTorch 会从尾部维度开始比较；每一维必须相等，或其中一个为 1，缺失维度视为 1。符合条件时，小维度会被逻辑扩展，而不是实际复制数据：

    m1 = torch.ones(2,3)
    m2 = torch.tensor([1,2,5])
    print(m1)
    print(m2)
    # m2=[1,2,5] 会被广播为 m2=[[1,2,5],[1,2,5]], 实际不会复制出一个新的矩阵，而是pytorch会按照广播机制去原始张量数据中取数据
    print(m1 + m2)

    m3 = torch.tensor([[10.1], [20.2]])
    print(m1 + m3)

    # 常见的 `[B, D]` 与 `[D]` 运算会对批次中每个样本使用同一组特征值。若广播报错，先把各操作数的 `shape` 写出来，逐维从右向左核对。
    # 即： 广播都是从内部开始的（高维）
    ...


def test_tensor7():
    # 4.3 矩阵乘法与线性代数

    m1 = torch.randn(2,3)
    m2 = torch.randn(3,4)
    # [2,3] @ [3,4] -> [2,4]
    # matmul 和 @ 效果相同，一个是符合运算一个是函数运算
    print(m1 @ m2)
    print(torch.matmul(m1, m2))

    # 注意：m1 * m2 是 逐个元素的相乘
    m11 = torch.randn(3,4)
    m21 = torch.randn(3,4)
    print(m11 * m21)

    # `*` 是逐元素乘法，`@`/`matmul` 是矩阵乘法。`matmul` 支持批量矩阵乘法，会对前置批次维度进行广播。常用线性代数函数还包括：
    m = torch.randn(3, 3)
    m_t = m.T
    inverse = torch.linalg.inv(m)  # 方阵可逆时求逆
    values, vectors = torch.linalg.eig(m)  # 特征值与特征向量
    norm = torch.linalg.norm(m)


    # torch.dot 只能用于两个一维 Tensor（向量）的点积运算，向量点积算的是标量
    # 两个一维向量进行点乘，结果为一个数
    m31 = torch.tensor([1,2,3])
    m32 = torch.tensor([4,5,6])
    # res = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    print(m31.dot(m32))



    ...

def test_tensor8():
    # torch.linalg 是 PyTorch 中的线性代数（Linear Algebra）模块
    # 用于处理矩阵和向量的数学运算，例如矩阵求逆、求解线性方程组、计算行列式、矩阵分解、求范数等。
    # 1. torch.linalg 常用操作
    # 函数	功能	示例用途
    # torch.linalg.inv()	求矩阵的逆	计算 \(A^{-1}\)
    # torch.linalg.det()	求矩阵的行列式	计算 \(\det(A)\)
    # torch.linalg.solve()	求解线性方程组	求 \(Ax=b\) 中的 \(x\)
    # torch.linalg.norm()	计算向量或矩阵的范数	计算向量长度
    # torch.linalg.eig()	求特征值和特征向量	分析矩阵性质
    # torch.linalg.svd()	奇异值分解	降维、数据分析
    # torch.linalg.qr()	QR 分解	数值计算
    # torch.linalg.matrix_rank()	计算矩阵的秩	判断矩阵的线性独立性

    # 矩阵求逆 ， a必须是可逆矩阵
    A = torch.tensor([[1,2],[3,4]], dtype=torch.float64)
    print(A)
    A_inv = torch.linalg.inv(A)
    print(A_inv)
    print(A @ A_inv)
    print(A_inv.matmul(A))

    # 求矩阵的行列式
    print(torch.linalg.det(A))

    # 求解线性方程组
    # x+2y=5, 3x+4y=11
    target = torch.tensor([5, 11], dtype=torch.float64)
    res = torch.linalg.solve(A, target)
    print(res)

    # 计算向量或矩阵的范数
    print(torch.linalg.norm(A))

    # 求特征值和特征向量
    print(torch.linalg.eig(A))

    # 计算矩阵的秩
    print(torch.linalg.matrix_rank(A))

    ...


def test_tensor9():
    # 4.4 聚合、统计与维度参数

    x = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float64)
    print(x)
    print(x.shape)
    print(x.size())
    print(x.min())
    print(x.max())
    # 返回最大值及其索引
    print(x.max(dim=1))

    # 越外面维度越低, 越外面dim对应的数值越小
    # `dim` 指被消去/聚合的轴
    print(x.sum(dim=0)) # 沿第 0 维聚合, [5, 7, 9]
    print(x.sum(dim=1)) # 沿第 1 维聚合, [12, 15]
    print(x.mean(dim=0)) # [2.5, 3.5, 4.5]
    print(x.mean(dim=1)) # [2, 5]
    print(x.sum(dim=1, keepdim=True)) # 保持维度不变

    ...


def test_tensor10():
    # 5. 形状变化

    x = torch.arange(24)

    #  5.1 `reshape` 与 `view`

    # `reshape` 在元素总数兼容时改变形状；如果内存布局允许，它可能返回共享存储的视图，否则可能复制。
    # `view` 要求底层内存布局兼容，不满足时会报错。
    # 一般优先使用 `reshape`；当明确需要视图语义时再用`view`。

    a = x.reshape(2, 3, 4)
    b = x.view(2, 3, 4)
    print(x)
    print(a)
    print(b)

    x = torch.arange(12).reshape(3, 4)
    print(x)
    print(x.reshape(2, 6))
    print(x.reshape(-1))
    print("="*30)

    # 5.2 增加、删除维度
    # 增加 unsqueeze
    # 删除 squeeze 挤压
    x = torch.arange(12).reshape(3, 4)
    # x 是 [3,4]
    x1 = x.unsqueeze(0) # x1 是 [1,3,4]
    x2 = x.unsqueeze(-1) # x2 是 [3,4,1]
    print(x1, x1.shape)
    print(x2, x2.shape)

    x3 = x1.squeeze(0)
    x4 = x2.squeeze(-1)
    print(x3, x3.shape)
    print(x4, x4.shape)


    ...

import numpy as np

def test_tensor11():
    # pytorch 与 numpy
    arr = np.array([1,2,3,4,5], dtype=np.float64)
    m1 = torch.tensor(arr)
    m2 = torch.from_numpy(arr)
    print(arr)
    print(m1)
    print(m2)
    arr[0] = 666
    # 因此可以看出，from_numpy 没有重新创建底层数组
    print(arr) # [666.,   2.,   3.,   4.,   5.]
    print(m1) # [1.,   2.,   3.,   4.,   5.]
    print(m2) # [666.,   2.,   3.,   4.,   5.]

    # 以下可以看出，numpy()之后的数组也是原来的数组
    arr2 = m1.numpy()
    arr2[1] = 999
    print(arr2)
    print(m1)
    print(m2)

    # 显示复制数据， 出来的两个数据，矩阵t_copy 和 数组array_copy 都是重新分配的内存
    t_copy = torch.tensor(arr)  # 复制
    array_copy = t_copy.detach().numpy().copy()


    ...


def test_tensor12():
    # 7. 自动求导 autograd
    # 自动求导会记录参与运算的张量操作，并按链式法则计算梯度。模型训练通常通过损失函数对参数求梯度，再由优化器更新参数。

    # requires_grad=True 表示需要跟踪与 x 有关的运算，以便后续计算梯度。
    x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    z = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    y = x ** 2 + 3 * x + z ** 2
    print(x)
    print(x.grad) # x.grad 用于保存反向传播后的梯度
    print(y) # tensor([10.], dtype=torch.float64, grad_fn=<AddBackward0>)， 10表示y(x)=x^2+3x，当x=2时的结果
    # 自动求导, 从y开始沿着计算过程反向传播，计算 y 对相关变量的梯度。执行y.backward()这个函数之后 x.grad 里面会保存 y函数对应x位置的导数
    y.backward()
    # y'(x)=2x+3, 当x=2时，y‘(2)=4+3=7
    print(x.grad) # tensor([7.], dtype=torch.float64),
    print(z.grad) # tensor([6.], dtype=torch.float64), y'(z)=2*z
    # 默认情况下，只有叶子张量的 `.grad` 会被保存。由其他运算得到的中间结果通常不是叶子张量；需要检查其中间梯度时可调用 `retain_grad()`。

    # PyTorch 默认会累积梯度，而不是每次调用 backward() 都自动清零。 因此训练时通常需要在下一轮反向传播前清理旧梯度。
    xx = x.grad.zero_()
    print(xx)
    print(x.grad)

    # 实际神经网络训练中是如下方式使用
    # optimizer = torch.optim.SGD([x, w], lr=0.01)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    print("="*30)

    # 7.2 向量输出与反向传播
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2
    print(x.sum())
    print(y)
    print(y.sum())
    # 这是因为 backward() 默认要求被反向传播的结果是标量（只有一个数的张量）。你的 y 是一个包含 3 个元素的一维张量，而 y.sum() 会把这 3 个元素相加，得到一个标量，因此可以直接调用 backward()。
    # z=y.sum(), 类似于 z=X1^2+X2^2+X3^2, 之后在对y.sum().backward()进行求导的时候就是对x1，x2，x3进行求偏导
    # y.backward(torch.ones_like(y))
    y.sum().backward()
    print(x.grad)  # tensor([2., 4., 6.])


    # 模拟梯度累加
    (3*x**2).sum().backward()
    print(x.grad)
    # 清空梯度
    # 以 `_` 结尾的许多操作会原地修改张量，例如 `add_()`、`zero_()`
    # x.grad.zero_()
    x.grad = None  # 也可使用 optimizer.zero_grad(set_to_none=True)
    print(x.grad)
    # y.sum().backward()
    # print(x.grad)

    # 对参与计算图的叶子张量做原地修改，可能破坏反向传播所需的历史值并触发错误。更新参数时应放在 `torch.no_grad()` 中，或交由优化器完成：
    w = torch.tensor(2.0, requires_grad=True)
    loss = w ** 2
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
    ...


def test_tensor13():
    # 8. dtype、设备与模型训练的基础流程
    # 8.1 CPU 与 GPU/MPS

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    x = torch.randn(2, 3).to(device)
    print(device)
    print(x)

    #  8.2 转换类型与设备
    x_float = x.float()
    print(x_float)
    x_long = x_float.long()
    print(x_long)
    x_cpu = x.cpu()
    print(x_cpu)

    ...

def current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def test_tensor14():
    # 9.1 排序、Top-K 与唯一值

    x = torch.tensor([5,6,3,9,1,8,2,1,3])
    print(x)
    print(x.sort())
    print(torch.topk(x, 3))
    print(torch.unique(x))

    # 在分类任务中，常用 `logits.topk(k, dim=-1)` 取得每个样本得分最高的类别。
    # torch.topk 常用于取分数较高的前 n 个
    x2 = torch.randn((3,4))
    print(x2)
    print(torch.topk(x2, 2))

    ...

def test_tensor15():
    # 9.2 `gather`：按索引从指定维度取值
    # `gather` 的输入和索引通常具有相同维度数；除指定维度外，索引在其他维度不能超过输入范围：

    scores = torch.tensor([[0.1, 0.8, 0.1],
                           [0.7, 0.1, 0.2],
                           [0.1, 0.1, 0.8],])
    print(scores)
    labels = torch.tensor([[1],[0],[2]])
    print(labels)
    # gather 根据指定的索引，从张量中提取对应位置的元素。
    print(scores.gather(1, labels))
    print(torch.gather(scores, 1, labels))

    # argmax
    print(scores.argmax(dim=1))
    # argmax() 返回每一行最大值所在的索引；而 gather() 根据你给定的索引，返回对应的实际数值。
    # argmax 给矩阵返回最大的索引，每行最大值所在的列索引
    # gather 给矩阵和索引返回对应位置的数据值，根据 labels 提取对应的分数

    ...

def test_tensor16():
    # 9.4 掩码填充与裁剪

    x = torch.tensor([1.0, float("nan"), 3.0])
    # 使用 0.0 来填充 nan 值
    clean = torch.nan_to_num(x, nan=0.0)
    print(x)
    print(clean)

    scores = torch.tensor([-2.0, 0.5, 4.0])
    # 小于0的设置为0，大于1的设置为1
    clipped = torch.clamp(scores, min=0.0, max=1.0)
    print(scores)
    print(clipped)
    print("="*30)

    logits = torch.randn(2, 4)
    padding_mask = torch.tensor([[False, False, True, True],
                                 [False, False, False, True]])
    print(logits)
    print(padding_mask)
    print(logits * padding_mask)
    print(logits * ~padding_mask)
    print(logits.masked_fill(padding_mask, 0.0))


    logits = torch.randn(2, 4, 5)  # [batch, length, classes]
    padding_mask = torch.tensor([[False, False, True, True],
                                 [False, False, False, True]])
    print(logits)
    print(padding_mask.unsqueeze(-1))
    print(logits.masked_fill(padding_mask.unsqueeze(-1), 0.0))

    ...


def test_tensor17():
    # 10. 保存、加载与检查张量

    # 保存训练模型（结构➕权重）
    # torch.save({"weights": model.state_dict(), "step": 100}, "checkpoint.pt")

    # 加载
    # checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=True)
    # model.load_state_dict(checkpoint["weights"])
    ...

def tets1():

    import asyncio

    type UserId = int
    type User = dict[str, str | int]

    a: UserId = 1
    qa: UserId = "1"
    print(a)
    print(qa)

    m: User = {}
    m["s"] = "ss"
    m["sd"] = 1
    m[1] = 123
    print(m)

    names = ["Alice", "Bob", "Charlie"]

    result = f"用户列表：{", ".join(names)}"
    print(result)

    async def fetch_user():
        await asyncio.sleep(0.1)
        raise ValueError("用户 ID 不合法")


    async def fetch_goods():
        await asyncio.sleep(0.1)
        raise ConnectionError("商品服务连接失败")


    async def fetch_orders():
        await asyncio.sleep(0.1)
        raise TimeoutError("订单服务请求超时")

    async def main():
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(fetch_user())
                tg.create_task(fetch_goods())
                tg.create_task(fetch_orders())

        except* ValueError as eg:
            print("用户参数异常：")
            for exc in eg.exceptions:
                print(f"  {type(exc).__name__}: {exc}")

        except* ConnectionError as eg:
            print("网络连接异常：")
            for exc in eg.exceptions:
                print(f"  {type(exc).__name__}: {exc}")

        except* TimeoutError as eg:
            print("请求超时异常：")
            for exc in eg.exceptions:
                print(f"  {type(exc).__name__}: {exc}")


    asyncio.run(main())


def test_view() -> None:
    # tensor.view
    # tensor.view() 的作用是：在不改变 Tensor 数据内容的前提下，改变 Tensor 的形状（shape）。

    x = torch.randn(2, 3)
    print(x)
    x_view = x.view(-1)
    print(x_view)

    re_x_view = x_view.reshape(3,2)
    print(x_view)
    print(re_x_view)


    z = torch.arange(24)
    print(z)
    print(z.view(2, 3, 4))
    print(z.view(4, 6))
    print(z.view(-1, 6))


    # 最常见的用途是：将多维数据展平，转换为全连接层需要的二维输入。
    # # 64 张图片，每张图片是 1 × 28 × 28
    x = torch.randn(64, 1, 28, 28)
    print(x.shape)

    # x.size(0) 为x的第0维即最外面的尺寸，即为64，
    # 将x变为[64，*]的数据，也就是批次不展开，后面其他的数据通通平铺开
    # 这样之后的x_view就可以输入到net里面了，因为全连接层 nn.Linear 通常接收形如 [batch_size, features] 的输入。
    x_view = x.view(x.size(0), -1)
    print(x_view.shape)

    import torch.nn as nn
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    x = torch.randn(64, 1, 28, 28)
    y = model(x)

    print(y.shape)



    ...