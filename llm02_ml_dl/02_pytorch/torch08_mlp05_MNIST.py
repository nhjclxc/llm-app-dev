# pytorch01_神经网络包nn和优化器optimizer相关教程.md
import time

import pytest
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def test_download_dataset():
    # 1. 准备数据集
    transform = transforms.ToTensor()

    train_dataset_no = MNIST('../data', train=True, download=True, transform=transform)
    test_dataset_no = MNIST('../data', train=False, download=True, transform=transform)

    train_dataset_no.download()
    test_dataset_no.download()

    print(len(train_dataset_no), type(train_dataset_no))

    image, label = train_dataset_no[0]
    # <PIL.Image.Image image mode=L size=28x28 at 0x109845250>，
    print(image, type(image))
    print(label)

    train_dataset = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform,
    )
    print(len(train_dataset), type(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )


# 2、定义多层感知机 mlp
class MNIST_MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 28*28=784, 将每一张图片铺开，就会有784个输入特征，因此输入层就有784个输入神经元
        # self.linear1 = nn.Linear(28*28, 256)
        # self.relu1 = nn.ReLU()
        # # 隐藏层里面逐步的将特征数变为10（10即为分类数）
        # self.linear2 = nn.Linear(256, 128)
        # self.relu2 = nn.ReLU()
        # # 输出层，输出特征定义为10，即表示10分类任务
        # # 注意输出层不需要激活函数
        # self.linear3 = nn.Linear(128, 10)
        # # 每一层的参数个数计算公式：输入特征个数*输出特征个数+输出特征。输入特征个数*输出特征个数是权重w的参数个数，输出特征是偏置项个数
        # # 总的参数个数=784*256+256 +256*128+128 +128*10+10

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 10),
        )
        ...

    def forward(self, x):
        # 前向传播

        # 将每张 28×28 图片展平为 784 个特征
        # x = torch.flatten(x, start_dim=1)
        # x = self.relu1(self.linear1(x))
        # x = self.relu2(self.linear2(x))
        # x = self.linear3(x)
        # return x

        x = self.flatten(x)
        return self.network(x)

# 3. 选择计算设备
def current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
    # device = torch.device("cuda") if torch.cuda.is_available() else (torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu"))
    # print(device)

def test_my_mlp():
    start = time.perf_counter_ns()
    # 1、加载数据
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    test_dataset = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
    )

    # 2、确定设备
    device = current_device()

    # 3、定义模型
    model = MNIST_MLPModel().to(device)


    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # 5、训练
    epochs = 5
    for epoch in range(epochs):
        # 开启训练模式
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0


        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 清空上一轮梯度
            optimizer.zero_grad()

            # 前向传播
            logits = model(images)

            # 计算损失
            loss = criterion(logits, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 统计 损失，精度，准确率
            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {avg_loss:.4f}, "
            f"Train Accuracy: {accuracy:.2%}"
        )

        ...
    # 6. 在测试集上评估模型
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct / total:.2%}")

    # 7. 保存模型参数
    torch.save(model.state_dict(), "mnist_model.pth")
    print("模型参数已保存")

    end = time.perf_counter_ns()
    print(f"总计耗时：{end - start} ms")

def test_total_params():
    device = current_device()
    model = MNIST_MLPModel().to(device)

    # 统计全部可训练参数
    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print("total_params = ", total_params)
    print("total_params2 = ", (784*256+256 +256*128+128 +128*10+10))

@pytest.fixture(scope="function")
def load_test_loader():

    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
    )
    return test_loader

def test_load_model_dict(load_test_loader):
    # 加载已训练的模型参数

    # 1. 创建模型结构
    device = current_device()
    model = MNIST_MLPModel().to(device)

    # 2. 加载已保存的模型参数
    state_dict = torch.load("mnist_model.pth", map_location=device, weights_only=True)

    # 3. 将参数加载到模型中
    model.load_state_dict(state_dict)

    # 4. 切换到评估模式
    model.eval()

    # 5. 测试模型能否正常预测
    with torch.no_grad():
        # 取一个批次的数据
        images, labels = next(iter(load_test_loader))
        images = images.to(device)
        labels = labels.to(device)
        print(images.shape, labels.shape)

        # 取一个数据
        x = images[0]
        y = labels[0]
        print(x.shape, y.shape, y)

        # 预测一个数据
        logits = model(x)
        predicted_digit = logits.argmax(dim=1).item()
        print("输出形状：", logits.shape)  # torch.Size([1, 10])
        print("预测数字：", predicted_digit)  # 0～9
        print("y：", y)  # 0～9

    pass
