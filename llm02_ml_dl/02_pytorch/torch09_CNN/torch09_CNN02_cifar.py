import argparse
import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

torch.manual_seed(23)


class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarCNN, self).__init__()
        self.features = nn.Sequential(
            # block1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=num_classes), # 32 /2 /2 /2 = 4
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 随机一个特征看看模型定义是否正确
# inx = torch.randn(128, 3, 32, 32)
# tmodel = CifarCNN()
# outx = tmodel(inx)
# print(outx.shape) # 预期是[128,10]，128是批量，10是分类数
# print(outx[:3, :])

def current_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_cifar10(root_path: str) -> (DataLoader, DataLoader, DataLoader):
    """

    :rtype: (DataLoader, DataLoader, DataLoader)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    full_train_set = datasets.CIFAR10(root_path, train=True, download=True, transform=transform_train)

    # train:valid = 9:1
    train_set_size = int(len(full_train_set) * 0.9)
    valid_set_size = len(full_train_set) - train_set_size
    train_set, valid_set = random_split(full_train_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(23))
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = datasets.CIFAR10(root_path, train=True, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    print("训练集数量:", len(train_set))
    print("验证集数量:", len(valid_set))
    print("测试集数量:", len(test_set))

    return train_loader, valid_loader, test_loader


def train(model: CifarCNN, device, train_loader, criterion, optimizer) -> (float, float):
    model.train()
    train_sample = 0
    train_loss_sum = 0
    train_correct = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # 统计损失
        size = images.shape[0]
        train_sample += size
        train_loss_sum += loss.item() * size

        # 统计正确率
        predictions = logits.argmax(dim=1)
        train_correct += (predictions == labels.long()).sum().item()

    train_loss = train_loss_sum / train_sample
    train_accuracy = train_correct / train_sample
    return train_loss, train_accuracy

def valid(model: CifarCNN, device, valid_loader, criterion) -> (float, float):
    model.eval()
    valid_sample = 0
    valid_loss_sum = 0
    valid_correct = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            size = images.shape[0]
            valid_sample += size
            valid_loss_sum += loss.item() * size
            predictions = torch.argmax(logits, dim=1)
            valid_correct += (predictions == labels.long()).sum().item()

    valid_loss = valid_loss_sum / valid_sample
    valid_accuracy = valid_correct / valid_sample
    return valid_loss, valid_accuracy

def _test(model: CifarCNN, device, test_loader) -> (float):
    # 测试阶段
    model.eval()
    test_accuracy = 0
    with torch.inference_mode():
        test_sample = 0
        test_correct = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            test_correct += (predictions == labels.long()).sum().item()
            test_sample += images.shape[0]
        test_accuracy = test_correct / test_sample
    return test_accuracy


def save_list(train_losses, valid_losses, train_accuracies, valid_accuracies, test_accuracy):
    import json
    data = {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "train_accuracies": train_accuracies,
        "valid_accuracies": valid_accuracies,
        "test_accuracy": test_accuracy,
    }

    # 保存到文件
    with open("./training_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main(dataset_path: str = "", epochs: int = 150, lr=0.01, net_test = False):

    path = "../../data/dl"
    device = current_device()
    print(f"当前使用的设备：{device}")
    print(f"开始加载数据")
    train_loader, valid_loader, test_loader = load_cifar10(dataset_path)
    print(f"加载数据结束")

    model = CifarCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    best_accuracy = -1.0
    best_model_state_dict = None

    print(f"开始训练模型")

    for epoch in range(epochs):

        # ==========================
        # 一、训练阶段
        # ==========================
        train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # ==========================
        # 二、验证阶段
        # ==========================
        valid_loss, valid_accuracy = valid(model, device, valid_loader, criterion)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        # ==========================
        # 三、保存最佳模型
        # ==========================
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_model_state_dict = copy.deepcopy(model.state_dict())

        # ==========================
        # 四、调整学习率
        # ==========================
        # 每一个 epoch 完成之后调整学习率
        scheduler.step()

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        if net_test:
            print(f"epoch: {epoch}, LR: {current_lr:.6f}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, "
                  f"train_accuracy: {train_accuracy:.4f}, valid_accuracy: {valid_accuracy:.4f}")
            break

        # ==========================
        # 五、打印训练结果
        # ==========================
        if (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch}, LR: {current_lr:.6f}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, "
                  f"train_accuracy: {train_accuracy:.4f}, valid_accuracy: {valid_accuracy:.4f}")

    print(f"训练模型结束")

    torch.save(best_model_state_dict, "./CifarCNN_best_model_state_dict.pth")
    print(best_model_state_dict.keys())
    print(f"best_accuracy: {best_accuracy}")

    # 测试阶段
    test_accuracy = _test(model, device, test_loader)

    # 保存训练的列表数据
    save_list(train_losses, valid_losses, train_accuracies, valid_accuracies, test_accuracy)

if __name__ == "__main__":

    print(1e-3)

    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--net_test", type=bool, default=False)

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数
    dataset_path = args.dataset_path
    epochs = args.epochs
    lr = args.lr
    net_test = args.net_test
    print("数据集路径:", dataset_path)
    print("训练轮数:", epochs)
    print("LR:", lr)
    print("net_test:", net_test)

    main(dataset_path, epochs, lr, net_test)

# mkdir -p /data/exec/cifar && cd /data/exec/cifar
# scp data/dl/cifar-10-python.tar.gz root@192.168.207.197:/data/exec/cifar
# scp 02_pytorch/torch09_CNN01_cifar.py root@192.168.207.197:/data/exec/cifar/cifar.py
# tar -xvzf cifar-10-python.tar.gz
# python3 -m pip install torch torchvision
# python3 cifar.py
# python3 torch09_CNN01_cifar.py --dataset_path="../../data/dl" --net_test=True

