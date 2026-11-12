# Autoencoder 自编码器：从重建原理到去噪、异常检测与 VAE

自编码器（Autoencoder，AE）是一类以“重建输入”为训练目标的神经网络。给定输入 `x`，模型先将它压缩成潜在表示 `z`，再从 `z` 重建出 `x_hat`：

```text
x (输入) -> Encoder 编码器 -> z (潜在向量) -> Decoder 解码器 -> x_hat (重建结果)
```

看起来标签就是输入本身，似乎模型只要“复制”即可；关键在于编码器和解码器之间通常设置了信息瓶颈（bottleneck），例如把 784 维图像压缩到 16 维。模型无法逐像素原样传递输入，只能学习数据中重复、稳定、可组合的结构，才能尽可能重建输入。

本文从零开始，以 MNIST 手写数字为主线，依次学习普通全连接自编码器、训练和可视化、去噪自编码器、卷积自编码器、异常检测与变分自编码器（VAE）。示例默认适用于 PyTorch 2.x。

```python
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

## 1. 学习路线与先修知识

| 阶段 | 核心问题 | 本文内容 |
| --- | --- | --- |
| 入门 | 自编码器在学习什么，为什么输入等于标签？ | 2-5 |
| 基础 | 如何搭建、训练和评估全连接 AE？ | 6-12 |
| 表示学习 | 潜在向量、瓶颈大小和重建质量有什么关系？ | 13-15 |
| 图像实践 | 为什么图像常使用卷积自编码器？ | 16-18 |
| 实用变体 | 如何去噪、做异常检测？ | 19-20 |
| 进阶 | VAE 如何把潜在空间变成可采样分布？ | 21-24 |

建议先掌握：Tensor shape、`nn.Module`、前向传播、反向传播、`DataLoader`、`Adam`、`MSELoss` 和 `BCEWithLogitsLoss`。自编码器是**无监督/自监督表示学习**方法：不依赖人工类别标签，但训练时仍有清晰监督信号，即原始输入本身。

## 2. 自编码器的基本原理

设输入样本为 `x`，编码器记作 `f_theta`，解码器记作 `g_phi`：

$$
z=f_\theta(x),\qquad \hat{x}=g_\phi(z)
$$

其中：

- `x`：原始输入。例如一张 MNIST 图片，shape 为 `[1, 28, 28]`。
- `z`：latent vector，潜在向量/编码。它是模型学习到的低维特征表示。
- `x_hat`：重建结果，shape 必须与 `x` 相同，才能逐元素计算重建误差。
- `theta`、`phi`：编码器和解码器的可学习参数。

训练目标是最小化输入和重建结果的距离：

$$
\mathcal{L}_{recon}=d(x,\hat{x})
$$

对于像素值已缩放到 `[0, 1]` 的连续灰度图，常使用均方误差（MSE）：

$$
\text{MSE}(x,\hat{x})=\frac{1}{D}\sum_{j=1}^{D}(x_j-\hat{x}_j)^2
$$

最小 shape 例子：

```python
x = torch.rand(4, 1, 28, 28)       # [batch=4, C=1, H=28, W=28]
z = torch.randn(4, 16)              # 每张图片压缩成 16 维表示
x_hat = torch.rand(4, 1, 28, 28)   # 重建结果必须同形状

loss = F.mse_loss(x_hat, x)
print(loss.shape)  # torch.Size([])，标量
```

## 3. 为什么“输入等于标签”不会让模型无意义地复制？

若网络容量无限、没有约束，它确实可能学习恒等映射 `x_hat = x`，这没有价值。因此 AE 需要至少一种约束。最常见的是**欠完备（undercomplete）瓶颈**：令 `z` 的维度远小于输入维度。

```text
MNIST 图像：28 x 28 = 784 维
        ↓ Encoder
潜在向量：16 维
        ↓ Decoder
重建图像：784 维
```

在 16 个数中无法保存每一个像素，模型必须发现训练数据的共同规律，例如数字的轮廓、笔画位置和粗细。其他常见约束有：

| 约束方法 | 做法 | 作用 |
| --- | --- | --- |
| 欠完备瓶颈 | 减小 `latent_dim` | 强制压缩信息 |
| 稀疏 AE | 对激活加稀疏正则 | 让少量单元激活 |
| 去噪 AE | 输入加入噪声，目标仍是干净图 | 学习稳定结构 |
| 正则化 AE | dropout、权重衰减等 | 减少记忆训练样本 |
| VAE | 约束潜在分布接近标准正态 | 得到连续、可采样空间 |

下面的代码说明“没有瓶颈”与“有瓶颈”的维度差异：

```python
input_dim = 28 * 28

identity_like = nn.Sequential(
    nn.Linear(input_dim, input_dim),
    nn.ReLU(),
    nn.Linear(input_dim, input_dim),
)

bottleneck_encoder = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 16),  # 784 -> 16，压缩发生在这里
)

x = torch.rand(8, input_dim)
print(identity_like(x).shape)         # [8, 784]
print(bottleneck_encoder(x).shape)    # [8, 16]
```

## 4. 编码器和解码器不是互为数学逆函数

“编码”和“解码”描述其角色，不代表 `decoder(encoder(x))` 是严格可逆变换。由于存在降维、非线性和有限参数，信息通常会丢失，`x_hat` 只是对 `x` 的近似。

以下例子中，编码器从 4 维压缩到 2 维，任何解码器都无法保证还原所有可能的 4 维向量：

```python
encoder = nn.Linear(4, 2)
decoder = nn.Linear(2, 4)

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
z = encoder(x)
x_hat = decoder(z)

print("z:", z.shape)          # [1, 2]
print("x_hat:", x_hat.shape)  # [1, 4]
print("reconstruction mse:", F.mse_loss(x_hat, x).item())
```

训练的目标不是保证每个任意输入都可逆，而是让模型在训练数据分布附近尽可能准确地重建。

## 5. 选择重建损失：MSE、BCE 与输出激活

损失函数和解码器最后一层必须匹配数据范围与概率假设。

| 输入/目标 | 常见输出层 | 常见损失 | 说明 |
| --- | --- | --- | --- |
| `[0, 1]` 连续灰度值 | `Sigmoid()` 或无激活 | `MSELoss` | 常见的图像重建起点 |
| 二值像素或 Bernoulli 假设 | 不加 sigmoid | `BCEWithLogitsLoss` | 数值更稳定，内部含 sigmoid |
| 已标准化为任意实数 | 无激活 | `MSELoss` / `L1Loss` | 不要再强加 sigmoid |
| 对离群重建误差更稳健 | 无激活或按范围选 | `L1Loss` / SmoothL1 | 误差的绝对值更敏感度均匀 |

MSE 版本：

```python
decoder_mse = nn.Sequential(
    nn.Linear(16, 784),
    nn.Sigmoid(),  # 将输出限制在 [0, 1]
)

target = torch.rand(8, 784)
prediction = decoder_mse(torch.randn(8, 16))
loss = nn.MSELoss()(prediction, target)
```

`BCEWithLogitsLoss` 版本。注意：使用它时模型最后一层**不能**再加 `Sigmoid()`：

```python
decoder_logits = nn.Linear(16, 784)  # 输出 logits，不加 Sigmoid
target = torch.randint(0, 2, (8, 784)).float()
logits = decoder_logits(torch.randn(8, 16))
loss = nn.BCEWithLogitsLoss()(logits, target)
```

## 6. 最小可运行：二维数据上的线性自编码器

先不使用图片。我们生成接近直线 `y = 2x` 的二维点，再把 2 维输入压缩到 1 维。训练后，1 维潜在变量会倾向于表示点在线上的位置。

```python
import matplotlib.pyplot as plt

n = 512
x1 = torch.linspace(-2, 2, n)
x2 = 2 * x1 + 0.15 * torch.randn(n)
points = torch.stack([x1, x2], dim=1)  # [512, 2]

plt.scatter(points[:, 0], points[:, 1], s=8)
plt.axis("equal")
plt.show()
```

定义模型。该例有意使用线性层，便于观察最小结构：

```python
class LinearAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 1)
        self.decoder = nn.Linear(1, 2)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

model = LinearAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()
dataset = TensorDataset(points)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

训练时，目标直接取 batch 输入：

```python
for epoch in range(200):
    model.train()
    total_loss = 0.0

    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        x_hat, _ = model(batch_x)
        loss = criterion(x_hat, batch_x)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)

    if (epoch + 1) % 50 == 0:
        print(f"epoch={epoch + 1}, mse={total_loss / len(dataset):.6f}")
```

检查压缩和重建 shape：

```python
model.eval()
with torch.inference_mode():
    x_hat, z = model(points.to(device))

print(z.shape)      # [512, 1]
print(x_hat.shape)  # [512, 2]
```

## 7. 为 MNIST 准备数据

MNIST 是 28 x 28 的单通道灰度图，`ToTensor()` 会把像素转为 `[0, 1]` 的 FloatTensor，因此适合使用 `Sigmoid + MSE` 的基础版本。

```python
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_set = datasets.MNIST(
    root="../../data",
    train=True,
    download=True,
    transform=transform,
)
test_set = datasets.MNIST(
    root="../../data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

images, labels = next(iter(train_loader))
print(images.shape)  # [256, 1, 28, 28]
print(labels.shape)  # [256]；AE 训练本身不会使用 labels
```

`DataLoader` 返回的第二项是数字类别，但普通自编码器不需要它。为了强调这一点，可以把数据集包一层，只返回图片：

```python
class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image

image_only_loader = DataLoader(ImageOnlyDataset(train_set), batch_size=256, shuffle=True)
images = next(iter(image_only_loader))
print(images.shape)  # [256, 1, 28, 28]
```

## 8. 全连接自编码器的结构与 shape

`nn.Linear` 处理二维输入 `[N, D]`，所以图片先需要展平为 `[N, 784]`，最后再恢复为 `[N, 1, 28, 28]`。这里使用 32 维潜在向量：

```python
class MLPAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),              # [N, 1, 28, 28] -> [N, 784]
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),  # [N, latent_dim]
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

model = MLPAutoencoder(latent_dim=32).to(device)
x = torch.rand(8, 1, 28, 28, device=device)
z = model.encode(x)
x_hat = model(x)
print(z.shape)      # [8, 32]
print(x_hat.shape)  # [8, 1, 28, 28]
```

这里 `forward()` 只返回 `x_hat`，符合训练循环最简接口；`encode()` 和 `decode()` 单独保留，便于后续获取潜在表示、插值或采样。

## 9. 完整训练循环：重建不是分类

分类训练通常使用 `labels`；AE 的 target 是 `images`。下面的函数同时返回按样本数加权的平均损失，避免最后一个较小 batch 被错误赋予完整 batch 的权重。

```python
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for images, _ in loader:
            images = images.to(device)
            reconstructed = model(images)
            loss = F.mse_loss(reconstructed, images)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / total_samples

model = MLPAutoencoder(latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

train_losses, test_losses = [], []
for epoch in range(20):
    train_loss = run_epoch(model, train_loader, optimizer)
    test_loss = run_epoch(model, test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"epoch={epoch + 1:02d}, train_mse={train_loss:.5f}, test_mse={test_loss:.5f}")
```

注意事项：

- 评估阶段必须使用 `model.eval()` 和 `torch.inference_mode()`；上面的函数已处理。
- `MSELoss` 的默认 reduction 是均值，loss 对每个 batch 已是所有像素、所有样本的平均值。
- `weight_decay` 是可选正则化，不是自编码器成立的必要条件。

## 10. 可视化原图与重建图

单一 loss 不能完整反映图像质量。应把原图和重建结果并排展示：

```python
import matplotlib.pyplot as plt

def show_reconstructions(model, loader, count=8):
    model.eval()
    images, _ = next(iter(loader))
    images = images[:count].to(device)

    with torch.inference_mode():
        reconstructed = model(images)

    images = images.cpu()
    reconstructed = reconstructed.cpu()
    fig, axes = plt.subplots(2, count, figsize=(count * 1.5, 3))

    for i in range(count):
        axes[0, i].imshow(images[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i, 0], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("recon")
    plt.tight_layout()
    plt.show()

show_reconstructions(model, test_loader)
```

观察时重点看：数字身份是否保留、笔画是否断裂、背景是否出现杂点、是否所有输出都变得相似。若所有重建图都近似同一个模糊数字，通常是模型容量/训练不足或损失设置有问题。

## 11. 保存与加载模型

建议保存 `state_dict`，而不是保存整个 Python 模型对象。加载时必须使用相同的模型结构：

```python
save_path = "../../data/output/mnist_mlp_autoencoder.pt"
torch.save(model.state_dict(), save_path)

loaded_model = MLPAutoencoder(latent_dim=32).to(device)
state_dict = torch.load(save_path, map_location=device)
loaded_model.load_state_dict(state_dict)
loaded_model.eval()
```

若需要恢复训练，还应保存 optimizer 状态、epoch 和最佳验证损失：

```python
checkpoint = {
    "epoch": 20,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "latent_dim": 32,
}
torch.save(checkpoint, "../../data/output/mnist_ae_checkpoint.pt")

checkpoint = torch.load("../../data/output/mnist_ae_checkpoint.pt", map_location=device)
model = MLPAutoencoder(latent_dim=checkpoint["latent_dim"]).to(device)
model.load_state_dict(checkpoint["model_state"])
```

## 12. 常见错误：shape、范围与错误的 target

### 12.1 忘记展平或恢复形状

`Linear(784, ...)` 不能直接接收 `[N, 1, 28, 28]`。用 `nn.Flatten()` 或 `x.flatten(1)`；输出时再 reshape。

```python
x = torch.rand(16, 1, 28, 28)
flat = x.flatten(start_dim=1)
print(flat.shape)  # [16, 784]

restored = flat.view(-1, 1, 28, 28)
print(restored.shape)  # [16, 1, 28, 28]
```

### 12.2 重建目标不能是类别标签

以下写法是错误的：`reconstructed` 是图像张量，而 `labels` 是类别编号，二者语义和 shape 都不同。

```python
images = torch.rand(8, 1, 28, 28)
labels = torch.randint(0, 10, (8,))
reconstructed = torch.rand(8, 1, 28, 28)

# 错误：F.mse_loss(reconstructed, labels)
correct_loss = F.mse_loss(reconstructed, images)
```

### 12.3 输出范围和数据范围不匹配

若输入经 `Normalize(mean=0.5, std=0.5)` 变为约 `[-1, 1]`，最后的 `Sigmoid()` 只能输出 `[0, 1]`，不合适。可以移除 Sigmoid，或把数据恢复到 `[0, 1]`。

```python
normalized_images = torch.rand(4, 1, 28, 28) * 2 - 1  # [-1, 1]
decoder = nn.Linear(16, 784)  # 不使用 Sigmoid
output = decoder(torch.randn(4, 16)).view(-1, 1, 28, 28)
loss = F.mse_loss(output, normalized_images)
```

## 13. 潜在维度 `latent_dim` 是什么

`latent_dim` 控制压缩强度。它不是越小越好，也不是越大越好：

| latent_dim | 典型现象 |
| --- | --- |
| 很小，例如 2 | 压缩强，重建会模糊，但便于二维可视化 |
| 中等，例如 16/32 | 常是 MNIST 的良好入门起点 |
| 很大，例如 256/512 | 重建容易变好，但表示可能接近记忆输入 |

可以统一训练不同维度并比较验证 MSE：

```python
latent_dims = [2, 8, 32, 128]
results = {}

for latent_dim in latent_dims:
    candidate = MLPAutoencoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-3)

    for _ in range(5):
        run_epoch(candidate, train_loader, optimizer)

    results[latent_dim] = run_epoch(candidate, test_loader)

print(results)
```

这不是严格实验，因为每个模型训练轮数较少；但可用于理解压缩率和重建误差之间的权衡。正式比较应固定随机种子、训练预算、数据划分和优化器设置。

## 14. 提取潜在表示并用于下游任务

训练完成后，编码器可把图片变成固定长度特征。下面提取整个测试集的 `z`，同时保留真实 label；label 只用于分析，不参与 AE 训练：

```python
def collect_latents(model, loader):
    model.eval()
    all_z, all_labels = [], []

    with torch.inference_mode():
        for images, labels in loader:
            z = model.encode(images.to(device))
            all_z.append(z.cpu())
            all_labels.append(labels)

    return torch.cat(all_z), torch.cat(all_labels)

test_z, test_labels = collect_latents(model, test_loader)
print(test_z.shape)       # [10000, latent_dim]
print(test_labels.shape)  # [10000]
```

如果 `latent_dim=2`，可以按类别着色观察表示空间：

```python
small_model = MLPAutoencoder(latent_dim=2).to(device)
# 先按照第 9 节训练 small_model
z2, labels = collect_latents(small_model, test_loader)

plt.figure(figsize=(7, 6))
scatter = plt.scatter(z2[:, 0], z2[:, 1], c=labels, s=4, cmap="tab10")
plt.colorbar(scatter)
plt.xlabel("z_0")
plt.ylabel("z_1")
plt.show()
```

类别不一定会完全分离，因为普通 AE 的目标是重建而非分类；但若数字结构被学习到，常能看到一定聚集趋势。

## 15. 潜在空间插值：观察表示是否连续

从两张图片得到 `z_a`、`z_b`，在二者之间线性插值得到一系列潜在向量，再解码为图片。若潜在空间较平滑，图像应逐渐变化而不是随机跳变。

```python
def interpolate(model, image_a, image_b, steps=8):
    model.eval()
    with torch.inference_mode():
        z_a = model.encode(image_a.unsqueeze(0).to(device))
        z_b = model.encode(image_b.unsqueeze(0).to(device))
        alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
        z = (1 - alphas) * z_a + alphas * z_b
        return model.decode(z).cpu()

images, _ = next(iter(test_loader))
interpolated = interpolate(model, images[0], images[1], steps=8)

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for ax, image in zip(axes, interpolated):
    ax.imshow(image[0], cmap="gray")
    ax.axis("off")
plt.show()
```

普通 AE 可以做插值，但不能保证从随机 `z ~ N(0, I)` 采样后能生成合理图片；这是 VAE 要解决的问题。

## 16. 为什么图像更适合卷积自编码器

全连接 AE 把图片展平，忽略了相邻像素之间的二维空间关系。卷积自编码器（Convolutional Autoencoder，CAE）直接处理 `[N, C, H, W]`，具有局部连接和权重共享，更适合图像。

编码器通常逐步降低空间尺寸、增加通道数；解码器通过转置卷积（`ConvTranspose2d`）或上采样恢复空间尺寸：

```text
[N, 1, 28, 28]
  -> Conv stride=2 -> [N, 32, 14, 14]
  -> Conv stride=2 -> [N, 64, 7, 7]
  -> ConvTranspose -> [N, 32, 14, 14]
  -> ConvTranspose -> [N, 1, 28, 28]
```

先检查一个卷积编码层的输出：

```python
encoder_layer = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
x = torch.rand(8, 1, 28, 28)
features = encoder_layer(x)
print(features.shape)  # [8, 32, 14, 14]
```

## 17. `ConvTranspose2d` 与输出尺寸

转置卷积常用于学习型上采样，但它不是普通卷积的严格数学逆。输出空间尺寸为：

$$
L_{out}=(L_{in}-1)S-2P+D(K-1)+\text{output_padding}+1
$$

其中 `S` 是 stride，`P` 是 padding，`D` 是 dilation。以下配置把 `7 x 7` 扩大为 `14 x 14`：

```python
up = nn.ConvTranspose2d(
    in_channels=64,
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1,
)
x = torch.randn(8, 64, 7, 7)
y = up(x)
print(y.shape)  # [8, 32, 14, 14]
```

若结果尺寸不符合预期，先打印每一层 shape；不要盲目靠 `view()` 修复。`output_padding` 只用于消除尺寸歧义，不是在输出上真正补零。

## 18. 完整卷积自编码器

该 CAE 输入和输出均为 `[N, 1, 28, 28]`，可直接复用第 9 节的训练函数：

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14 -> 7
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14 -> 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

conv_ae = ConvAutoencoder().to(device)
images = torch.rand(8, 1, 28, 28, device=device)
reconstructed = conv_ae(images)
print(reconstructed.shape)  # [8, 1, 28, 28]

optimizer = torch.optim.Adam(conv_ae.parameters(), lr=1e-3)
for epoch in range(10):
    train_loss = run_epoch(conv_ae, train_loader, optimizer)
    test_loss = run_epoch(conv_ae, test_loader)
    print(f"epoch={epoch + 1}, train={train_loss:.5f}, test={test_loss:.5f}")
```

CAE 的潜在表示在这里仍是 `[N, 64, 7, 7]` 的特征图，而非单个向量。这是合理的：压缩也可以保留低分辨率空间结构。若需要固定维度向量，可在 encoder 后再加 `Flatten + Linear`，decoder 前再用 `Linear + Unflatten`。

## 19. 去噪自编码器：输入有噪声，目标是干净样本

去噪自编码器（Denoising Autoencoder，DAE）训练时故意破坏输入 `x` 得到 `x_tilde`，但重建目标仍是干净 `x`：

$$
z=f_\theta(\tilde{x}),\qquad \hat{x}=g_\phi(z),\qquad \mathcal{L}=d(\hat{x},x)
$$

这迫使模型学习数字轮廓等稳定特征，而非逐像素复制噪声。添加高斯噪声的函数：

```python
def add_gaussian_noise(images, std=0.35):
    noisy = images + std * torch.randn_like(images)
    return noisy.clamp(0.0, 1.0)

clean = torch.rand(4, 1, 28, 28)
noisy = add_gaussian_noise(clean)
print(noisy.min().item(), noisy.max().item())  # 仍在 [0, 1]
```

训练循环中的关键差异是模型输入为 `noisy`，loss 的 target 为 `clean`：

```python
def train_denoising_epoch(model, loader, optimizer, noise_std=0.35):
    model.train()
    total_loss, total_samples = 0.0, 0

    for clean, _ in loader:
        clean = clean.to(device)
        noisy = add_gaussian_noise(clean, std=noise_std)
        reconstructed = model(noisy)
        loss = F.mse_loss(reconstructed, clean)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clean.size(0)
        total_samples += clean.size(0)

    return total_loss / total_samples

denoising_model = ConvAutoencoder().to(device)
optimizer = torch.optim.Adam(denoising_model.parameters(), lr=1e-3)
for epoch in range(10):
    loss = train_denoising_epoch(denoising_model, train_loader, optimizer)
    print(f"epoch={epoch + 1}, denoising_mse={loss:.5f}")
```

不要写成 `mse(reconstructed, noisy)`；那样网络会学会保留噪声，而不是去噪。

## 20. 用重建误差做异常检测

异常检测的基本假设：只用正常样本训练 AE，模型会擅长重建正常模式；遇到不符合该模式的异常输入，重建误差往往更高。

例如只用 MNIST 的数字 `0` 训练：

```python
normal_digit = 0
normal_indices = [
    i for i, (_, label) in enumerate(train_set)
    if label == normal_digit
]
normal_train_set = torch.utils.data.Subset(train_set, normal_indices)
normal_loader = DataLoader(normal_train_set, batch_size=128, shuffle=True)

normal_ae = ConvAutoencoder().to(device)
optimizer = torch.optim.Adam(normal_ae.parameters(), lr=1e-3)
for _ in range(10):
    run_epoch(normal_ae, normal_loader, optimizer)
```

必须按**每个样本**计算误差，不能只计算一个 batch 平均值：

```python
def reconstruction_errors(model, loader):
    model.eval()
    errors, labels = [], []

    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device)
            reconstructed = model(images)
            # 从 [N, 1, 28, 28] 聚合为 [N]
            per_sample_mse = (reconstructed - images).pow(2).flatten(1).mean(dim=1)
            errors.append(per_sample_mse.cpu())
            labels.append(batch_labels)

    return torch.cat(errors), torch.cat(labels)

errors, labels = reconstruction_errors(normal_ae, test_loader)
print(errors.shape)  # [10000]
```

阈值应该用正常验证集的误差分布确定，例如 99 分位数，而不是随意选择：

```python
normal_errors = errors[labels == normal_digit]
threshold = torch.quantile(normal_errors, q=0.99)
predicted_anomaly = errors > threshold
print("threshold:", threshold.item())
```

重建误差高不必然证明样本异常。复杂背景、分布漂移、模型欠拟合都可能抬高误差；应使用独立标注数据评估 precision、recall、ROC-AUC 或 PR-AUC。

## 21. 普通 AE 为什么不一定能随机生成图片

普通 AE 只要求训练样本经过 encoder 得到的 `z` 能被 decoder 重建。它没有规定这些 `z` 在空间中应服从什么分布；不同训练样本对应的编码可能散落在不连续区域。

因此，下面从标准正态随机取的 `z` 不一定在训练编码分布附近：

```python
ae = MLPAutoencoder(latent_dim=32).to(device)
random_z = torch.randn(16, 32, device=device)

with torch.inference_mode():
    samples = ae.decode(random_z)

print(samples.shape)  # shape 正确，但图像质量没有理论保证
```

变分自编码器（VAE）通过把每个输入的潜在表示建模成分布，并用 KL 散度把这些分布拉向 `N(0, I)`，使随机采样更有意义。

## 22. VAE 原理：编码的是分布参数

普通 AE 编码器输出一个确定的 `z`；VAE 编码器输出高斯分布的均值 `mu` 和对数方差 `logvar`：

$$
q_\phi(z|x)=\mathcal{N}(\mu(x),\operatorname{diag}(\sigma^2(x)))
$$

然后从该分布采样潜在变量。直接采样会阻断梯度，因此采用重参数化技巧：

$$
z=\mu+\sigma\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
$$

采样函数：

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

mu = torch.zeros(4, 8)
logvar = torch.zeros(4, 8)  # log(variance)=0，方差为 1
z = reparameterize(mu, logvar)
print(z.shape)  # [4, 8]
```

`logvar` 比直接预测方差更方便：指数可保证方差为正，且数值优化通常更稳定。

## 23. 一个 MNIST VAE 模型

以下 VAE 使用 MLP，与第 8 节的全连接 AE 便于对照。decoder 输出 logits，因此后面使用 `BCEWithLogitsLoss`：

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),  # logits，不加 Sigmoid
        )

    def encode(self, x):
        h = self.backbone(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode_logits(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar

vae = VAE(latent_dim=16).to(device)
images = torch.rand(8, 1, 28, 28, device=device)
logits, mu, logvar = vae(images)
print(logits.shape, mu.shape, logvar.shape)  # [8, 784], [8, 16], [8, 16]
```

## 24. VAE 损失：重建损失加 KL 散度

VAE 的损失包含两部分：

$$
\mathcal{L}=\mathcal{L}_{recon}+\beta\,D_{KL}(q_\phi(z|x)\parallel \mathcal{N}(0,I))
$$

对对角高斯，单样本 KL 项为：

$$
D_{KL}=-\frac{1}{2}\sum_i(1+\log\sigma_i^2-\mu_i^2-\sigma_i^2)
$$

代码实现。两项都用 batch 的均值，保证 `beta` 的含义稳定：

```python
def vae_loss(logits, images, mu, logvar, beta=1.0):
    target = images.flatten(start_dim=1)
    reconstruction = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
    reconstruction = reconstruction / images.size(0)

    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = kl_per_sample.mean()
    total = reconstruction + beta * kl
    return total, reconstruction, kl

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
for epoch in range(10):
    vae.train()
    for images, _ in train_loader:
        images = images.to(device)
        logits, mu, logvar = vae(images)
        loss, recon_loss, kl_loss = vae_loss(logits, images, mu, logvar, beta=1.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(
        f"epoch={epoch + 1}, total={loss.item():.2f}, "
        f"recon={recon_loss.item():.2f}, kl={kl_loss.item():.2f}"
    )
```

VAE 通常比普通 AE 的重建更模糊，因为它不仅优化重建，也要求潜在空间规整；其优势是可以从先验分布采样生成样本：

```python
vae.eval()
with torch.inference_mode():
    z = torch.randn(16, 16, device=device)
    generated = torch.sigmoid(vae.decode_logits(z)).view(-1, 1, 28, 28)

print(generated.shape)  # [16, 1, 28, 28]
```

## 25. 训练诊断与排错清单

训练 AE 时，先检查这些不变量：

```python
images, _ = next(iter(train_loader))
images = images.to(device)
model = ConvAutoencoder().to(device)
output = model(images)

assert output.shape == images.shape
assert torch.isfinite(output).all()
assert torch.isfinite(F.mse_loss(output, images))
print("shape and finite checks passed")
```

常见症状与排查方向：

| 症状 | 常见原因 | 首先检查 |
| --- | --- | --- |
| loss 极大 | 像素值范围不一致、忘记标准化 | `images.min()/max()`、输出激活 |
| loss 为 NaN | 学习率过大、数据含 NaN/Inf | `torch.isfinite`、降低 lr |
| 输出全黑/全灰 | 训练不足、输出激活饱和、容量不足 | 原图/重建图、梯度、lr |
| loss 很低但视觉很差 | 过度平均、MSE 的模糊效应 | 可视化、尝试 L1/结构损失 |
| 训练好、测试差 | 过拟合、数据分布不同 | train/test 曲线、正则化 |
| `size mismatch` | encoder/decoder 空间尺寸算错 | 每层打印 shape |

在卷积模型中插入 shape 调试：

```python
class DebugConvAE(ConvAutoencoder):
    def forward(self, x):
        print("input:", x.shape)
        z = self.encoder(x)
        print("latent feature map:", z.shape)
        x_hat = self.decoder(z)
        print("output:", x_hat.shape)
        return x_hat

debug_model = DebugConvAE()
_ = debug_model(torch.rand(2, 1, 28, 28))
```

## 26. 从零到一练习路线

按以下顺序完成，避免一开始直接进入 VAE 或异常检测：

1. **二维线性 AE**：运行第 6 节，理解 `x -> z -> x_hat` 和重建 loss。
2. **MNIST MLP AE**：完成第 7-10 节，画出原图与重建图。
3. **压缩实验**：将 `latent_dim` 改为 `2、8、32、128`，记录 MSE 和可视化结果。
4. **潜在表示**：训练 `latent_dim=2` 的模型，完成第 14 节散点图。
5. **卷积 AE**：实现第 18 节，对比 MLP 与卷积 AE 的图像清晰度。
6. **去噪 AE**：给输入加入高斯噪声，验证 target 始终是干净图。
7. **异常检测**：只用某一类样本训练，使用每样本误差和验证阈值。
8. **VAE**：实现重参数化、KL 损失与随机采样，理解“重建”和“生成”的区别。

一个最小自测代码：

```python
def smoke_test_autoencoder():
    model = MLPAutoencoder(latent_dim=8)
    x = torch.rand(3, 1, 28, 28)
    x_hat = model(x)
    loss = F.mse_loss(x_hat, x)

    assert x_hat.shape == x.shape
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    print("AE smoke test passed, loss:", loss.item())

smoke_test_autoencoder()
```

## 27. 总结

普通自编码器的主线是：

```text
输入 x
  -> 编码器得到 z
  -> 解码器得到 x_hat
  -> 用 x_hat 与 x 计算重建损失
  -> 反向传播同时更新 Encoder 和 Decoder
```

最重要的工程规则是：

- 明确每一层的 Tensor shape，最终必须满足 `x_hat.shape == x.shape`。
- 让输出激活、损失函数与输入数据范围匹配。
- 自编码器训练时 target 是输入，不是类别 label。
- 用可视化检查重建质量，不能只看单个 loss。
- 普通 AE 适合压缩和表征学习；去噪 AE 适合恢复干净结构；CAE 更适合图像；VAE 才具备受约束的随机生成能力。
