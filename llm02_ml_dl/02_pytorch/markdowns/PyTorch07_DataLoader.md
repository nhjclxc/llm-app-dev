# PyTorch Dataset 与 DataLoader：从 0 到 1

训练模型之前，要先解决一个实际问题：怎样稳定地从数据集中取出样本，并按 batch 交给模型？`Dataset` 负责定义“一个样本是什么、如何读取”；`DataLoader` 负责按批次取样、打乱、组合和（可选）并行加载。

本文从一个 Python 列表开始，逐步学习内存 Tensor、map-style Dataset、自定义 Dataset、DataLoader 参数、训练循环、变长序列、流式数据和性能排错。示例默认：

```python
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
```

## 1. 学习地图

| 阶段 | 学习目标 | 内容 |
| --- | --- | --- |
| 入门 | 理解样本、批次和迭代 | 2–3 |
| 基础使用 | 组合 Tensor，配置 DataLoader | 4–5 |
| 实战 | 编写 Dataset、加载 CSV/文件 | 6–7 |
| 训练集成 | 划分数据并完成训练循环 | 8–9 |
| 进阶 | 自定义 collate、IterableDataset | 10–11 |
| 提效排错 | workers、pin memory、随机性、常见问题 | 12–15 |

## 2. 为什么需要 DataLoader

假设有 1000 条样本。最直接的做法是一次性把全部数据交给模型，但数据可能无法全部放进内存或显存；同时模型训练通常按小批次（mini-batch）进行。手动切片也能实现批处理，但需要自己处理打乱、最后一个不足整批的样本、复杂样本拼接、并行读取等细节。

DataLoader 让数据按需组成 batch：

```python
samples = [(torch.tensor([1.0, 2.0]), 0),
           (torch.tensor([3.0, 4.0]), 1),
           (torch.tensor([5.0, 6.0]), 0)]
loader = DataLoader(samples, batch_size=2, shuffle=False)

for features, labels in loader:
    print(features.shape, labels.shape)
# torch.Size([2, 2]) torch.Size([2])
# torch.Size([1, 2]) torch.Size([1])
```

默认 collate 会把可堆叠的样本字段合并成批次张量。上例最后一个 batch 只有一条记录。

## 3. Dataset 与 DataLoader 的分工

可以把二者理解为数据读取的两个层次：

```text
数据源 -> Dataset[index] 读取/转换一条样本 -> DataLoader 取样并组合 batch -> 训练循环 -> 模型
```

- `Dataset`：描述数据集及单个样本的读取方式。典型接口是 `__len__()` 和 `__getitem__(index)`。
- `DataLoader`：包裹一个 Dataset，产生可迭代的 batch。控制 `batch_size`、`shuffle`、`num_workers`、`collate_fn` 等。
- 训练循环：取出 batch，将数据放到合适设备，调用模型和损失函数。

DataLoader 本身通常不负责把整个数据集永久放进显存，也不替模型做训练或自动搬运数据到 GPU。

## 4. 最简单的数据集：`TensorDataset`

当特征和标签已经是内存中的 Tensor，使用 `TensorDataset`：

```python
features = torch.randn(100, 4)  # 100 条样本，每条 4 个特征
labels = torch.randint(0, 3, (100,))
dataset = TensorDataset(features, labels)

print(len(dataset))       # 100
print(dataset[0])         # (第 0 条特征, 第 0 条标签)
```

所有传给 `TensorDataset` 的 Tensor 第 0 维长度必须相同。索引同一个位置会得到一条样本的 tuple：

```python
x0, y0 = dataset[0]
print(x0.shape)  # [4]
print(y0.shape)  # 标量 Tensor，即 shape []
```

然后交给 DataLoader：

```python
loader = DataLoader(dataset, batch_size=16, shuffle=True)
for x_batch, y_batch in loader:
    print(x_batch.shape, y_batch.shape)
    break
# [16, 4] [16]
```

分类标签通常是类别索引 `long`；回归目标通常是浮点数。Dataset 返回的数据类型应与模型和损失函数的约定一致。

### 4.1 `TensorDataset` 适用范围

适合小型数据或已经整体载入内存的数据。若原始图片/音频很大，把所有数据预先读为一个大 Tensor 会占用大量 RAM；此时更适合自定义 Dataset，按索引读取单个文件或记录。

## 5. 创建 DataLoader：常用参数

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)
```

| 参数 | 作用 | 常见建议 |
| --- | --- | --- |
| `dataset` | 数据来源，通常是 Dataset | 先确认单条样本返回内容 |
| `batch_size` | 每个 batch 的样本数 | 从显存/内存能承受的值开始 |
| `shuffle` | 每轮迭代前打乱索引 | 训练集通常 `True`；验证/测试通常 `False` |
| `num_workers` | 并行加载 worker 进程数 | 初始用 `0` 排错，再按环境调优 |
| `drop_last` | 是否丢弃最后一个不完整 batch | 训练可按需要开启；评估通常关闭 |
| `collate_fn` | 将若干样本合成 batch 的函数 | 默认规则不适合变长/复杂样本时自定义 |
| `pin_memory` | 将 CPU batch 放入页锁定内存 | CUDA 训练可尝试配合异步拷贝 |
| `persistent_workers` | epoch 间保留 worker 进程 | 仅 `num_workers > 0` 时有效，适合多轮训练 |
| `prefetch_factor` | 每个 worker 预取 batch 数 | worker 开启时可调，增加 RAM 占用 |
| `generator` | 控制 sampler 的随机数生成器 | 需要可复现的 shuffle 时使用 |

`batch_size` 通常不能与自定义 `batch_sampler` 一起指定。`shuffle` 与 `sampler` 也不能同时使用，因为 sampler 已经决定索引顺序。

### 5.1 `batch_size` 和最后一个 batch

若数据量不能被 batch size 整除，默认 `drop_last=False` 会保留最后的少量样本。`drop_last=True` 会直接丢弃它们：

```python
loader = DataLoader(dataset, batch_size=32, drop_last=True)
```

训练中有些结构（例如特定 BatchNorm 情形或要求固定 batch 形状的代码）可能偏好完整 batch；评估时一般不要丢弃数据。务必意识到 `drop_last=True` 会让一部分训练样本在该轮没有参与训练，验证/测试也会造成统计不完整。

### 5.2 训练、验证、测试的 shuffle

- 训练集通常打乱，让每轮 batch 组成和样本顺序不同。
- 验证集和测试集通常不需要打乱，便于结果复现和对应样本。
- 若使用 `sampler` 或分布式 sampler，采样顺序由 sampler 控制；不要再设 `shuffle=True`。

`shuffle=True` 每次新的 DataLoader 迭代通常产生新的随机顺序，不代表样本被重复抽取；标准 map-style 数据集的随机采样器默认仍是对索引做一轮排列。

## 6. 自定义 map-style Dataset

map-style Dataset 能通过索引读取样本，最常用的方式是继承 `Dataset` 并实现：

- `__len__()`：返回数据集样本总数。
- `__getitem__(index)`：返回一条样本，通常是 Tensor、tuple、list 或 dict。

```python
class ToyDataset(Dataset):
    def __init__(self, count=100):
        self.features = torch.randn(count, 4)
        self.labels = (self.features[:, 0] > 0).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

dataset = ToyDataset(100)
print(dataset[0])
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

对 map-style Dataset，DataLoader/sampler 通过索引决定取哪些样本。这样可以洗牌、按子集采样，也能用 `Subset` 划分数据。

### 6.1 Dataset 返回什么

一条样本可以返回 tuple：

```python
return feature_tensor, label_tensor
```

也可以返回带名称的 dict：

```python
return {"image": image, "label": label, "path": str(path)}
```

默认 collate 可以递归处理不少 Tensor、数字、列表、dict 结构，但 batch 内对应 Tensor 通常需要相同形状。字符串等字段通常会被收集成列表；自定义结构应先实际打印一个 batch 验证。

## 7. 从文件或表格读取数据

自定义 Dataset 常见工作是在 `__init__` 中建立文件清单/索引，在 `__getitem__` 中按需读取样本：

```python
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.transform = transform
        if len(self.image_paths) != len(self.labels):
            raise ValueError("图片路径和标签数量必须相同")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 使用 with 确保文件在转换完成后关闭
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
])
dataset = ImageFolderDataset(image_paths, labels, transform)
```

此例需要额外安装并导入 Pillow、torchvision。图片在 `__getitem__` 里按需打开，`transform` 把图片转换为 Tensor。确保所有图片经变换后 shape 相同，默认 collate 才能 stack 成 `[N, C, H, W]`。

### 7.1 CSV 数据的思路

表格数据通常先用 CSV 读取库解析行，保存文件路径/偏移或轻量元数据，在 `__getitem__` 里取得一条记录，并将特征转换成 float Tensor、分类标签转换成 long Tensor。要点：

- 特征列与标签列要明确分开，避免标签泄漏到输入。
- 缺失值处理、类别编码、标准化参数只在训练集拟合，再用于验证和测试集。
- 小型表格可以一次读到内存后用 `TensorDataset`；大型表格再按行/块读取。
- 如果在每次 `__getitem__` 中重新读取并解析整个 CSV，速度会很慢；初始化阶段做适当索引或预处理。

## 8. 划分训练集、验证集与测试集

对于 map-style Dataset，可使用 `random_split`：

```python
from torch.utils.data import random_split

dataset = ToyDataset(1000)
train_size = 800
valid_size = 100
test_size = len(dataset) - train_size - valid_size
generator = torch.Generator().manual_seed(42)
train_set, valid_set, test_set = random_split(
    dataset, [train_size, valid_size, test_size], generator=generator
)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
```

`random_split` 返回 `Subset`，包含原始 Dataset 和对应索引，不会把每条样本复制一份。分层分类、按时间切分、按用户/病人分组切分时，不应盲目随机切分；要按任务防止类别比例失衡、时间穿越或同一主体泄漏。

### 8.1 先切分，再拟合预处理

标准化均值/方差、缺失值填充值、词表等统计量需要从训练集拟合，再复用于验证和测试集。先对全量数据算统计量再划分会泄漏验证/测试信息。

如果同一个 Dataset 的 transform 含随机增强，`random_split` 的几个 Subset 会共享这个 Dataset 和 transform。训练集一般需要随机增强，而验证/测试一般只做确定性预处理；可创建不同的 Dataset 实例或设计清晰的数据管线，确保验证样本不会被随机增强。

## 9. 把 DataLoader 接入训练循环

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

    train_loss = total_loss / total_samples
```

一个 batch 通常是 CPU Tensor；模型和输入必须位于兼容设备。常用写法是在训练循环中把每个 batch 移至设备，而不是将整个 Dataset 放进 GPU。

验证时：

```python
model.eval()
valid_loss_sum = 0.0
valid_samples = 0
with torch.inference_mode():
    for features, labels in valid_loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        n = features.size(0)
        valid_loss_sum += loss.item() * n
        valid_samples += n
valid_loss = valid_loss_sum / valid_samples
```

以上按样本数累计适用于普通样本级、batch mean loss。若是有 `ignore_index` 的 token/像素损失、加权损失或变长数据，应该按有效元素/有效权重累计分子和分母。

## 10. 自定义 `collate_fn`

默认 collate 通常会对同一字段的 Tensor 执行 stack，因此要求它们形状相同。图像尺寸已统一的分类数据通常不需要自定义；变长文本、可变数量的目标框或带元信息的复杂 batch 则可能需要。

### 10.1 变长序列 padding

```python
from torch.nn.utils.rnn import pad_sequence

def collate_sequences(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, lengths, labels

# Dataset 中每个样本可返回：一维 token Tensor、标量标签 Tensor
loader = DataLoader(
    sequence_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_sequences,
)
```

collate 函数输入是一个样本列表，输出就是训练循环拿到的 batch。长度信息可用于 pack、attention mask 或忽略 padding 的损失。padding token 是否应计入 loss 要由损失函数的 `ignore_index` 或自定义 mask 处理。

### 10.2 自定义 batch 结构

collate 也可将 tuple 合成 dict，或合并检测任务中的不等长标注。保持输出格式清晰、易于 `.to(device)`：

```python
def collate_dict(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    paths = [item["path"] for item in batch]
    return {"images": images, "labels": labels, "paths": paths}
```

字符串路径留在 CPU 通常即可，不必递归移动整个 dict 中的所有内容。

## 11. IterableDataset：流式、队列与无限数据源

map-style Dataset 能按索引读取；若数据来自实时流、数据库游标、大型压缩文件，无法方便地建立随机索引，可继承 `IterableDataset` 并实现 `__iter__()`：

```python
from torch.utils.data import IterableDataset

class NumberStream(IterableDataset):
    def __init__(self, count):
        self.count = count

    def __iter__(self):
        for i in range(self.count):
            yield torch.tensor(float(i))

stream = NumberStream(10)
loader = DataLoader(stream, batch_size=4)
for batch in loader:
    print(batch)
```

IterableDataset 没有通用随机索引，因此通常不能使用 `shuffle=True` 或标准 `random_split`。epoch 长度可能未知，`len(loader)` 也可能无法获得准确结果。

### 11.1 多 worker 的重复数据问题

IterableDataset 使用多个 worker 时，每个 worker 都会复制 Dataset 实例。如果不按 worker 分片，每个 worker 可能从流的开头读取，导致数据重复。可用 `get_worker_info()` 分配互不重叠的数据范围：

```python
from torch.utils.data import get_worker_info

class ShardedNumberStream(IterableDataset):
    def __init__(self, count):
        self.count = count

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            start, end = 0, self.count
        else:
            per_worker = (self.count + info.num_workers - 1) // info.num_workers
            start = info.id * per_worker
            end = min(start + per_worker, self.count)
        for i in range(start, end):
            yield torch.tensor(i)
```

真实流式数据要结合数据源本身做分片（例如文件、分区、消息队列 partition），还要考虑 worker 数量变化、断点恢复和样本去重。

## 12. 加载性能与设备传输

### 12.1 从 `num_workers=0` 开始

`num_workers=0` 表示主进程同步读取，调试最简单，异常堆栈也通常更直观。确认 Dataset 正确后，尝试 2、4 等 worker 数并实际测吞吐量。worker 过多会增加进程启动、CPU/RAM 使用和存储竞争，未必更快。

macOS/Windows 常使用 spawn 启动 worker。脚本入口要保护主程序，避免每个子进程重新执行训练启动代码：

```python
def main():
    dataset = ...
    loader = DataLoader(dataset, num_workers=2)
    # 训练代码

if __name__ == "__main__":
    main()
```

Notebook 中多 worker 可能受运行环境限制；遇到卡住或序列化错误，先退回 `num_workers=0`。

### 12.2 `pin_memory` 与 `non_blocking`

CUDA 训练可尝试让 DataLoader 固定 CPU 内存，再以非阻塞方式传输：

```python
loader = DataLoader(dataset, batch_size=64, pin_memory=True, num_workers=2)
for x, y in loader:
    x = x.to("cuda", non_blocking=True)
    y = y.to("cuda", non_blocking=True)
```

这不是对所有设备都有效的加速开关，会增加 pinned RAM；应在 CUDA 场景 benchmark。MPS/CPU 工作流不需要照搬 CUDA 参数。

### 12.3 预取和持久 worker

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
)
```

`persistent_workers=True` 避免每个 epoch 结束后重启 worker，适合多 epoch 训练，但 worker 常驻会占资源。`prefetch_factor` 增大时队列里预先准备的 batch 更多，可能改善 GPU 等待数据，也会增加内存使用。先测数据是否真是瓶颈，再调这些参数。

## 13. 随机性与可复现

训练通常要打乱样本，但实验也常需要固定随机种子。可用 `generator` 固定 DataLoader sampler 的初始随机状态：

```python
generator = torch.Generator().manual_seed(42)
loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)
```

多 worker 中 Python、NumPy 或其他库的随机变换也需要初始化 worker 随机种子。PyTorch 给每个 worker 提供种子，可将其传给其他随机数生成器：

```python
import random
import numpy as np

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

generator = torch.Generator().manual_seed(42)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=generator,
)
```

该代码要求安装 NumPy。固定种子能提高可复现性，但不同设备、库版本、并行调度和非确定性算子仍可能导致结果不同。

## 14. Sampler 与特殊采样

`sampler` 决定 Dataset 索引的取样顺序；适用于加权抽样、指定子集、分布式训练等场景。

```python
from torch.utils.data import WeightedRandomSampler

# train_labels 与 dataset 索引一一对应；每个样本按其类别频率的倒数加权
class_counts = torch.bincount(train_labels)
sample_weights = 1.0 / class_counts[train_labels].float()
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),  # 每轮抽取的样本数
    replacement=True,
)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

`replacement=True` 允许一轮内重复抽到同一样本、也可能漏掉其他样本；它不同于简单 shuffle。类别不平衡时，过采样与损失加权是不同策略，可能重复强调少数样本，要在验证集比较。

分布式训练一般使用 `DistributedSampler`，并在每个 epoch 调用 `sampler.set_epoch(epoch)`，使各 epoch 的 shuffle 顺序改变；使用 sampler 时由 sampler 控制采样，DataLoader 不再设置 `shuffle=True`。

## 15. 常见问题排查

| 现象 | 常见原因 | 检查方式 |
| --- | --- | --- |
| `stack expects each tensor to be equal size` | 一个 batch 中样本 shape 不同 | 统一 resize，或自定义 `collate_fn` 做 padding |
| 特征和标签数量不一致 | Dataset 的索引映射或源文件清单不匹配 | 检查 `len(dataset)` 与首尾样本 |
| 标签 dtype 不符合 loss | 分类标签不是 long，BCE 标签不是 float | 打印每个 batch 的 dtype |
| 训练极慢 | Python 读取/解码成为瓶颈，或 worker 设置不合适 | 对比 `num_workers=0/2/4`，测样本吞吐 |
| worker 报错难读 | 多进程隐藏了原始异常 | 暂时设置 `num_workers=0` 复现 |
| 程序重复启动或递归创建 worker | spawn 环境未保护入口 | 加 `if __name__ == "__main__":` |
| 每个 epoch 样本重复 | IterableDataset 多 worker 未分片 | 用 `get_worker_info()` 或数据源分区 |
| 验证数量少于预期 | `drop_last=True` 或 sampler 只覆盖部分样本 | 检查 loader 参数及 `len(loader)` |
| 数据泄漏/验证指标异常高 | 划分前拟合预处理、重复主体跨集合 | 检查切分单位和预处理拟合范围 |
| GPU 利用率低 | 数据准备或 CPU 到 GPU 拷贝是瓶颈 | profiler/计时；再评估 worker 与 pin memory |

开始训练前先检查一个 batch：

```python
batch = next(iter(train_loader))
print(type(batch))
print(batch[0].shape, batch[0].dtype)
print(batch[1].shape, batch[1].dtype)
```

如果 Dataset 返回 dict，就分别检查每个字段。还要确认标签值范围、图像通道顺序、序列长度、设备和是否有 NaN/Inf。

## 16. 一个从数据到训练的最小完整例子

下面生成合成的二分类数据，使用 TensorDataset/DataLoader 完成一个训练 epoch：

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
features = torch.randn(200, 4)
labels = (features[:, 0] + features[:, 1] > 0).long()
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

model.train()
for x_batch, y_batch in loader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    optimizer.step()

print("最后一个 batch loss:", loss.item())
```

这个例子只演示数据流，不是完整的评估实验。真实项目还要划分验证集、记录整个 epoch 的平均 loss/指标、保存最佳模型，并确保训练和推理使用相同的确定性预处理。

## 17. 学习练习与验收

1. 用 `TensorDataset` 封装 100 条特征和标签，检查 `dataset[0]` 与 batch shape。
2. 写一个返回 dict 的自定义 Dataset，在 DataLoader 中检查默认 collate 的输出结构。
3. 对 map-style Dataset 使用固定 generator 做随机划分，并创建 train/valid/test loader。
4. 编写图片或 CSV Dataset，把文件读取、预处理与标签转换封装到 `__getitem__`。
5. 用变长 token 序列实现 `collate_fn` padding，验证 lengths 和 padding mask。
6. 分别测试 `num_workers=0` 和多个 worker，观察耗时、内存和异常行为。
7. 实现 IterableDataset 多 worker 分片，验证一个 epoch 没有重复或遗漏样本。

完成标准：能清楚解释单条样本由谁读取、batch 由谁组成、索引/打乱由谁控制、数据何时传到设备，以及每个 DataLoader 参数为什么这样设置。

## 18. 常用 API 速查

| API | 用途 |
| --- | --- |
| `torch.utils.data.Dataset` | 实现 map-style 数据集 |
| `torch.utils.data.IterableDataset` | 实现流式/迭代式数据集 |
| `torch.utils.data.DataLoader` | 批处理与迭代加载 |
| `torch.utils.data.TensorDataset` | 将多个同长度 Tensor 按首维组合 |
| `torch.utils.data.random_split` | 随机切分 map-style Dataset |
| `torch.utils.data.Subset` | 用索引构造数据子集 |
| `torch.utils.data.WeightedRandomSampler` | 按样本权重抽样 |
| `torch.utils.data.DistributedSampler` | 分布式训练的索引分片与采样 |
| `torch.utils.data.get_worker_info` | 获取 IterableDataset worker 信息 |

## 19. 关键结论与资料

- `Dataset` 负责“如何得到一个样本”；`DataLoader` 负责“如何取样并组成 batch”。
- 小型 Tensor 数据用 `TensorDataset` 快速开始；文件型数据通常按需读取并实现自定义 Dataset。
- 训练集一般 shuffle，验证/测试一般不 shuffle；`drop_last` 会丢数据，需明确选择。
- 默认 collate 要求可堆叠字段的 shape 一致；变长样本用 padding 和自定义 `collate_fn`。
- 从 `num_workers=0` 调试，性能参数要以吞吐、内存和硬件实测为准。
- IterableDataset 多 worker 必须设计数据分片，否则会重复消费数据。

参考资料：

- [PyTorch `torch.utils.data` 文档](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch Dataset 与 DataLoader 教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch DataLoader API](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
