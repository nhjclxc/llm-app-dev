
# PyTorch02_神经网络包nn和优化器optimizer相关教程.md

import torch
from torch import nn


# nn.Module官方文档：https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html
# https://blog.csdn.net/qq_27825451/article/details/90550890



def test_nn1() -> None:
    # 1. `torch.nn` 和 `torch.optim` 的职责

    # 典型训练流程由四部分构成：`nn.Module` 定义模型，损失函数衡量预测误差，autograd 计算梯度，optimizer 根据梯度更新参数。

    # model.train()
    # for inputs, targets in train_loader:
    #     optimizer.zero_grad(set_to_none=True)
    #     outputs = model(inputs)
    #     loss = criterion(outputs, targets)
    #     loss.backward()
    #     optimizer.step()

    # 优化器不会自动计算损失，也不会读取数据；它只管理给定参数并根据参数的 `.grad` 执行更新。

    ...


class MyModel(nn.Module):
    def __init__(self):
        # `__init__` 中定义子层

        # `super().__init__()` 用于初始化模块的参数和子模块注册机制。
        # super().__init__()  # 第一句话，调用父类的构造函数
        super(MyModel, self).__init__()  # 第一句话，调用父类的构造函数

        # 特征数=4*2+2
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        # `forward` 中描述数据流
        return self.linear(x)


def test_nn2() -> None:
    # 2. `nn.Module` 基础

    model = MyModel()
    x = torch.randn(3, 4) # [batchs, feature]
    y = model(x)  # 用 model(x)，不要直接调用 model.forward(x)
    print(y, y.shape)  # [3, 2]

    ...

# 使用 nn.Module 实现 y=2x+3模型
class MyPreModle(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
    def forward(self, x):
        return self.linear(x)

    pass



# 3. 选择计算设备
def current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# 如何使用nn.Module定义一个y=2x+3的模型
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义线性层：y = wx + b
        # 一个输入，一个输出
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)



def norm(input: torch.Tensor) -> torch.Tensor:
    # 实现 Min-Max 归一化，将数据缩放到 [0, 1]。使得数据相差不会太远
    x_min = input.min()
    x_max = input.max()

    # 当最大最小值相等的时候，确保不会出现除0异常
    denominator = x_max - x_min
    if denominator == 0:
        return torch.zeros_like(input)

    x_norm = (input - x_min) / (x_max - x_min)
    return x_norm

def test_norm() :
    x = torch.tensor([10.0, 20.0, 30.0, 40.0])
    x_norm = norm(x)
    print(x_norm)

def test_norm2() :
    x = torch.tensor([10.0, 10.0, 10.0, 10.0])
    x_norm = norm(x)
    print(x_norm)


def test_nn3() -> None:

    # 创建模型
    model = LinearModel()

    # 准备数据 y=2x+3 的训练数据
    batch_size = 10

    x_train_arr = [i+1 for i in range(1024)]
    x_train = torch.tensor(x_train_arr, dtype=torch.float32).reshape(-1, 1)
    x_train = norm(x_train) # 对数据进行归一化，防止梯度爆炸
    y_train = 2 * x_train + 3
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True
    )

    x_test_arr = [-i-1 for i in range(10)]
    x_test = torch.tensor(x_test_arr, dtype=torch.float32).reshape(-1, 1)
    x_test = norm(x_test)
    y_test = 2 * x_test +3
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=batch_size
    )

    # 定义损失和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    target_loss = 1e-5
    epochs = 500
    for epoch in range(epochs):
        # 开启训练模式
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            # 清空旧梯度
            optimizer.zero_grad()
            # 前向传播
            y_output = model(x)
            # 计算损失
            loss = criterion(y_output, y)
            # 反向传播
            loss.backward()
            # 更新梯度
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(train_loader)
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Loss: {avg_loss:.6f}"
            )
            # 当损失达到预期的时候，提前停止训练
            # if avg_loss < target_loss:
            #     print(f"损失达到目标，提前结束：Epoch {epoch + 1}")
            #     break

    # 查看训练结果
    print("权重w：", model.linear.weight.detach().numpy())
    print("偏置b：", model.linear.bias.detach().numpy())
    print("最终损失loss", loss.item())

    # 预测
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for x, y in test_loader:
            y_predict= model(x)
            loss = criterion(y_predict, y)
            total_loss += loss.item()

            print(f"x = {x} 时的预测结果：{y_predict}")
        avg_test_loss = total_loss / len(test_loader)
        print(f"测试集平均损失：{avg_test_loss:.6f}")

    # 保存模型➕参数
    torch.save(model, "../data/model/LinearModel.pth")

    # 仅保存参数
    torch.save(model.state_dict(), "../data/model/LinearModel-dict.pth")


    pass

def test_load_linear_model() -> None:
    # 加载模型和参数
    model = torch.load(
        "../data/model/LinearModel.pth",
        map_location="cpu",
        weights_only=False
    )
    do_eval(model)

def test_load_linear_model_dict() -> None:
    # 仅加载模型参数

    # 定义模型
    model = LinearModel()
    # 加载参数
    model.load_state_dict(torch.load("../data/model/LinearModel-dict.pth", map_location=current_device()))

    do_eval(model)

def do_eval(model: nn.Module) -> None:
    # 模型预测
    x_pred = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0, 5000.0], dtype=torch.float32).reshape(-1, 1)
    y_pred = 2 * x_pred + 3

    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        total_loss = 0
        correct = 0
        for x,y in zip(x_pred, y_pred):
            y_output = model(x)
            loss = criterion(y_output, y)
            total_loss += loss.item()
            flag = y.item() == y_output.item()
            print(f"预测值: {y.item()}, 真实值: {y_output.item()}，是否正确: {flag}")
            if flag:
                correct += 1
        print(f"总损失：{total_loss:.2f}, 预测正确数：{correct}")
        pass
    pass


def test_model_params() -> None:
    # 注册参数，输出模型参数
    model = MyModel()

    # nn.Linear(4, 2) 表示输入 4 个特征，输出 2 个特征。
    # PyTorch 的全连接层权重形状为：[out_features,in_features]，即[2,4]

    # 2 * 4 个权重，2 个偏置项
    print(list(model.named_parameters()))
    print()
    print(model.state_dict().keys())

    pass

# 定义多个全连接层fc
# 使用 使用 `ModuleList` 或 `Sequential` 来连接
class MultFCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, 10),
        )

        self.layer2 = nn.ModuleList([
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 10),
        ])

    def forward(self, x):
        return self.layer(x)


def test_nn5() -> None:
    # 2.3 训练模式和评估模式

    # 控制模型的运行模式
    # model.train()  # 切换到训练模式
    # model.eval()  # 切换到评估模式

    # 控制是否记录梯度
    # 训练时候的模型评估
    # with torch.no_grad(): # 暂停梯度记录
    # 模型输出之后，实际使用实际预测用的模式，纯推理/部署模型的情况下使用
    # with torch.inference_mode(): # 更彻底地关闭梯度相关机制


    # train() / eval()：决定模型层怎么运行。
    # no_grad()：不记录梯度，适用范围比较灵活。
    # inference_mode()：面向纯推理的进一步优化。
    # 正常训练通常使用 train()；
    # 训练的验证阶段使用 eval() 加 no_grad();
    # 纯推理或部署阶段使用 eval() 加 inference_mode()。

    ...


# 3. 常用网络层

class LinearModel2(nn.Module):
    def __init__(self):
        super().__init__()
        # linear=wx+b
        self.linear = nn.Linear(5,2)
        # linear2=wx, 没有偏置项
        self.linear2 = nn.Linear(5,2, bias=False)
    def forward(self, x):
        return self.linear(x)

def test_nn6() -> None:
    # 3.1 全连接层 `nn.Linear`

    # `nn.Linear(in_features, out_features)` 对最后一维执行线性变换。输入最后一维必须等于 `in_features`：

    model = LinearModel2()
    x = torch.randn(3, 5, dtype=torch.float32)
    print(x, x.shape)
    # 输入是[3,5]经过fc之后变为[3,2]
    y = model(x)
    print(y, y.shape)


    x2 = torch.randn(2, 3, 5, dtype=torch.float32)
    print(x2, x2.shape)
    # 输入是[2,3,5]经过fc之后变为[2,3,2]
    # 输入的第一维度可以理解为是一个批量2
    y2 = model(x2)
    print(y2, y2.shape)


    # 3.2 卷积、池化和展平
    # 二维图像卷积通常使用 `[N, C, H, W]` 布局：

    # 维度编号是从外到内排列的，0开始
    # 以x2为例
    # 第 0 维：最外层
    # 2 个元素，每个元素的形状是 [3, 2]
    # 第 1 维：中间层
    # 每个元素包含 3 行，每行有 2 个数
    # 第 2 维：最内层
    # 每行有 2 个数


    # start_dim=0 表示从最外层维度开始，
    flatten = nn.Flatten(start_dim=0)
    print(flatten(x).shape)  # [5, 3] -> [15],
    print(flatten(x2).shape)  # [2, 5, 3] -> [30], 从第 0 维开始，将所有维度合并：2*5*3
    # start_dim=1 表示保留最外层维度，从第二个维度开始展平。
    flatten2 = nn.Flatten(start_dim=1)  # 保留批次维度
    print(flatten2(x2).shape)  # [2, 5, 3] -> [2, 15], 保留第0维，从第1维开始合并[2, 5*3]

    # 如果是MNIST 手写数字图片的批次数据形状通常是：[批次数，通道数，特征数1，特征数2]
    # 那么铺开就是 nn.Flatten(start_dim=1) 表示保留第一个批次数，从2二个批次数开始展开，即[批次数, 通道数*特征数1*特征数2]


    ...


def test_nn7() -> None:
    # https://blog.csdn.net/ting_qifengl/article/details/130418573
    # 4.1 激活函数
    # 默认使用relu系列
    relu = nn.ReLU()
    leaky_relu = nn.LeakyReLU(0.01)
    p_relu = nn.PReLU()
    r_relu = nn.RReLU()

    # 多分类 Sigmoid
    sigmoid = nn.Sigmoid()


    gelu = nn.GELU()
    tanh = nn.Tanh()
    ...



def init_weights(module):
    # 初始化神经网络中全连接层（nn.Linear）的权重和偏置，让模型在开始训练之前拥有合适的参数初始值。
    if isinstance(module, nn.Linear):
        # 使用 Xavier 均匀分布初始化全连接层的权重
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



def test_nn8() -> None:
    # 自定义权重初始化函数
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    # 对模型中的子模块应用初始化函数
    model.apply(init_weights)
    ...
