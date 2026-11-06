# CNN 卷积神经网络：从卷积核到图像分类

卷积神经网络（Convolutional Neural Network，CNN）是一类利用局部连接和参数共享处理网格数据的神经网络。图像是最常见的例子：卷积层从局部区域提取边缘、纹理和形状等特征，后续层逐步组合这些特征，最终完成分类、检测或分割。

本文依次介绍图像张量、卷积核、步长、填充、特征图、池化、感受野，以及使用 PyTorch 搭建和检查一个小型 CNN。示例默认：

```python
import torch
from torch import nn
```

## 1. 学习路线与先修知识

| 阶段 | 学习目标 | 内容 |
| --- | --- | --- |
| 入门 | 理解图像张量和卷积直觉 | 2–3 |
| 基础 | 掌握卷积、步长、填充和输出尺寸 | 4–6 |
| 结构 | 理解池化、特征图和感受野 | 7–9 |
| 实战 | 搭建分类 CNN 并训练 | 10–12 |
| 进阶 | 理解常见模块、初始化和排错 | 13–16 |

建议先熟悉 Tensor 的 shape、`nn.Module`、损失函数、autograd 和 DataLoader。CNN 的核心是张量维度变换：输入的通道数、空间尺寸和 batch 维必须与每层的配置匹配。

## 2. 图像在 PyTorch 中的形状

PyTorch 的二维卷积通常使用 NCHW 格式：

```text
[N, C, H, W]
 N  Number	样本数量（Batch Size）
 C  Channel	通道数（如 RGB 的 3 个通道）
 H  Height	图像高度
 W  Width	图像宽度
```

例如，一批 32 张 RGB 图片、每张高 64 宽 64：

```python
images = torch.randn(32, 3, 64, 64)
print(images.shape)  # torch.Size([32, 3, 64, 64])
```

灰度图通常是 `[N, 1, H, W]`，RGB 图通常是 `[N, 3, H, W]`。单张图像作为 `Conv2d` 输入时可以是 `[C, H, W]`；批量输入通常是四维。不要把 NHWC（`[N, H, W, C]`）误当作 NCHW。

## 3. 卷积的直觉

二维卷积核（kernel / filter）是一个可学习的小权重矩阵。它在输入上按位置滑动，每到一个位置，就将局部输入与核权重逐元素相乘并求和，再加上偏置，得到输出特征图在该位置的值。

以单通道的 `3 x 3` 核为例，它每次查看输入的一个 `3 x 3` 局部区域。训练时，核中的权重通过反向传播学习；不同卷积核可以响应不同的模式，例如某个方向的边缘或某种纹理。具体学到什么由数据和训练目标决定，并非人工预先指定。

卷积具有两个重要特点：

- **局部连接**：每个输出位置只依赖输入的一个局部窗口。
- **参数共享**：同一组核权重在所有空间位置重复使用，因此模型能在不同位置检测相似模式，参数量也比全连接地连接整张图小得多。

## 4. `nn.Conv2d` 与卷积核

常用接口：

```python
nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
)
```

| 参数 | 含义 |
| --- | --- |
| `in_channels` | 输入通道数，必须与输入 Tensor 的 C 维相同 |
| `out_channels` | 卷积层输出的特征图（Feature Maps）数量，也是输出特征图的通道数 |
| `kernel_size` | 核的高和宽；整数表示两边相同，也可传 `(kh, kw)` |
| `stride` | 核每次移动的步长，整数或 `(sh, sw)` |
| `padding` | 输入边界补的像素数，整数或 `(ph, pw)`；也支持部分字符串模式 |
| `dilation` | 核元素之间的间距，用于扩大有效核覆盖范围 |
| `groups` | 输入、输出通道分组数；影响通道连接方式 |
| `bias` | 是否为每个输出通道学习偏置 |

最常见的普通卷积令 `groups=1`，每个输出通道都能组合全部输入通道。若输入是 3 通道、输出是 16 通道、核大小为 3：

```python
# 3个输入通道，16个输出特征图，卷积核大小是3
conv = nn.Conv2d(3, 16, kernel_size=3)
x = torch.randn(8, 3, 32, 32)
y = conv(x)
print(y.shape)  # [8, 16, 30, 30]，stride=1、padding=0
```

权重形状是 `[out_channels, in_channels / groups, kh, kw]`；偏置形状是 `[out_channels]`。上例权重形状为 `[16, 3, 3, 3]`。当卷积核不为正方形时，可以分别设置高度与宽度：

```python
conv = nn.Conv2d(3, 12, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
```

## 5. 步长 `stride`

步长表示卷积核在输入上移动多少格。`stride=1` 会逐格扫描；`stride=2` 每次移动两格，空间尺寸通常会缩小。横纵步长可以不同：`stride=(2, 1)`。

步长增大可以降低后续计算量并缩小特征图，但会跳过更多位置，空间细节也会减少。步长为 1 是常见的特征提取设置；使用步长 2 的卷积常作为下采样方式之一。

## 6. 填充 `padding` 与输出尺寸

不填充（`padding=0`）时，卷积窗口不能越过图像边界，输出尺寸会变小。填充通常在四周补零，使边缘位置也能参与计算，并控制输出空间大小。

对于每个空间维度，卷积输出大小为：

$
L_{out}=\left\lfloor\frac{L_{in}+2P-D(K-1)-1}{S}+1\right\rfloor
$

其中 `L_in` 是输入长度，`K` 是核大小，`S` 是步长，`P` 是 padding，`D` 是 dilation。高度和宽度分别计算。例如输入 `32 x 32`，核 `3 x 3`、stride=1、padding=0、dilation=1：

```text
floor((32 + 0 - 2 - 1) / 1 + 1) = 30
```

所以输出是 `30 x 30`。若设置 `padding=1`，输出为 `32 x 32`。对奇数核且 stride=1，`padding=(K-1)//2` 常用于保持空间尺寸。

`padding="same"` 在支持该配置的 PyTorch 版本中可让 stride=1 时输出空间尺寸与输入相同；它通常不支持大于 1 的 stride。精确支持条件请查当前 PyTorch 版本文档。`padding="valid"` 表示不填充。学习公式时，显式使用整数 padding 更容易核对尺寸。

```python
def conv_out_size(length, kernel_size, stride=1, padding=0, dilation=1):
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

print(conv_out_size(32, kernel_size=3, stride=2, padding=1))  # 16
```

## 7. 特征图与通道

卷积层输出的每个通道是一张特征图（feature map）。`out_channels=16` 意味着产生 16 张特征图，而不是把输入变成 16 张图片；batch 维仍然保留。

```python
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
x = torch.randn(4, 3, 28, 28)
y = conv(x)
print(y.shape)  # [4, 16, 28, 28]
```

空间尺寸由核、步长、填充和 dilation 决定；通道数由 `out_channels` 决定；样本数 `N` 不变。增加通道通常能让网络表示更多种特征，但计算量和参数量也会增加。

## 8. `nn.MaxPool2d` 最大池化

最大池化在局部窗口中取最大值，通常用于降低特征图的高宽：

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]])
print(pool(x))  # 每个 2x2 窗口保留最大值 4
```

常用参数包括 `kernel_size`、`stride`、`padding`、`dilation` 和 `ceil_mode`。省略 stride 时，默认等于 kernel_size；`kernel_size=2, stride=2` 常把高宽各缩小一半。最大池化不包含可学习权重，也不改变通道数。

```python
x = torch.randn(8, 16, 28, 28)
y = nn.MaxPool2d(kernel_size=2, stride=2)(x)
print(y.shape)  # [8, 16, 14, 14]
```

池化输出尺寸的基本计算与卷积相似（池化没有可学习卷积核权重）；padding、dilation 和 ceil_mode 会影响边界及取整。也可用 stride 大于 1 的卷积下采样。最大池化强调局部最大响应，平均池化计算窗口均值；二者带来的信息保留方式不同，需按任务选择。

## 9. 感受野 `receptive field`

某个深层特征图位置所依赖的原始输入区域称为它的感受野。浅层卷积的感受野较小，通常捕捉局部模式；连续堆叠卷积或下采样后，深层位置能整合更大区域的信息。

对于 stride=1、dilation=1 的连续卷积，两个 `3 x 3` 卷积层的理论感受野是 `5 x 5`，三个是 `7 x 7`，而不是每层都仍为 `3 x 3`。更一般地，设第 `l` 层核大小为 `k_l`、步长为 `s_l`、dilation 为 `d_l`，从输入开始令 `r_0=1, j_0=1`：

$
r_l = r_{l-1} + (k_l-1)d_l j_{l-1},\qquad j_l=j_{l-1}s_l
$

`r_l` 是理论感受野边长，`j_l` 是相邻该层位置在原图上的间隔（有效步幅）。二维且横纵参数不同的情况需分别对高度和宽度计算。池化层也会扩大感受野，计算时可把其窗口大小和 stride 纳入同类递推。padding 会影响边缘处有效输入范围；上述数值描述理论最大覆盖范围，不代表该范围内每个像素对输出的实际影响相等。

扩大感受野有助于结合更大范围的上下文，但可能损失定位细节。分类通常能接受逐步下采样；分割、关键点等密集预测任务则需要保留或恢复空间分辨率。

## 10. 组合卷积模块

常见基础模块是卷积、归一化、激活函数的组合：

```python
block = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
)

x = torch.randn(8, 3, 32, 32)
y = block(x)
print(y.shape)  # [8, 16, 16, 16]
```

`BatchNorm2d(num_features)` 中 `num_features` 是通道数；它在训练时使用 batch 统计量并维护运行统计量，在 `eval()` 时使用运行统计量。小 batch、分布式或推理场景要考虑其行为。`bias=False` 常与紧随其后的 BatchNorm 搭配，因为归一化层已有可学习偏移；这是常见配置，不是硬性规则。

## 11. 一个小型图像分类 CNN

以下模型接受 `[N, 1, 28, 28]` 的灰度图，输出 10 类 logits。两次 `2 x 2` 池化将空间尺寸从 28 缩到 14 再缩到 7。

```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [N, 16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [N, 16, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # [N, 32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [N, 32, 7, 7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)  # 原始 logits，不在这里做 softmax

model = SmallCNN(num_classes=10)
x = torch.randn(4, 1, 28, 28)
logits = model(x)
print(logits.shape)  # [4, 10]
```

多分类训练通常把 logits 直接传给 `nn.CrossEntropyLoss()`，标签形状为 `[N]`、dtype 为 `torch.long`。训练时不要先对 logits 做 softmax；推理时如需概率，再使用 `torch.softmax(logits, dim=1)`。

如果输入尺寸可能变化，固定写死 `32 * 7 * 7` 会不匹配。可考虑自适应池化：

```python
self.pool = nn.AdaptiveAvgPool2d((1, 1))
# forward 中：x = self.pool(x); x = torch.flatten(x, 1)
# 后续 Linear 的输入特征数就等于最后一个卷积层的通道数
```

`AdaptiveAvgPool2d((1, 1))` 把任意可接受的空间尺寸汇总为每通道一个值，不要求输入高宽固定为训练时的某个数值。

## 12. 完整训练步骤

典型的分类训练循环仍是标准 PyTorch 流程，CNN 只改变模型结构和输入形状：

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for images, labels in train_loader:  # images: [N, C, H, W]
    images = images.to(device)
    labels = labels.to(device, dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)            # [N, num_classes]
    loss = criterion(logits, labels)  # labels: [N]
    loss.backward()
    optimizer.step()
```

验证时调用 `model.eval()` 并在 `torch.inference_mode()` 或 `torch.no_grad()` 中计算；这会关闭梯度记录，也让 Dropout、BatchNorm 等模块使用推理行为。不要忘记把模型和输入放在同一设备。

## 13. 参数量与计算量直觉

普通卷积的可学习参数量（包含 bias）为：

```text
out_channels * (in_channels / groups) * kernel_height * kernel_width
+ (out_channels if bias else 0)
```

例如 `Conv2d(3, 16, 3)` 有 `16 * 3 * 3 * 3 + 16 = 448` 个参数。权重共享让参数量不依赖输入图像的高和宽；但图像越大，卷积需要计算的空间位置越多，计算量仍显著增加。

使用 `groups=in_channels` 且 `out_channels` 是 `in_channels` 的整数倍时，可以做深度卷积（depthwise convolution）；它对各输入通道分别卷积，不混合通道。深度可分离卷积通常再接一个 `1 x 1` pointwise 卷积混合通道，以较低计算量构造高效模块。

## 14. 常见 CNN 层与用途

| 层 | 作用 | 常见注意点 |
| --- | --- | --- |
| `nn.Conv2d` | 学习局部空间特征 | 输入通道、输出 shape、padding 和参数量 |
| `nn.ReLU` / `nn.GELU` | 引入非线性 | 仅堆叠线性卷积仍等价于线性变换 |
| `nn.MaxPool2d` | 局部最大值下采样 | 丢失空间细节；默认 stride 通常等于窗口 |
| `nn.AvgPool2d` | 局部平均下采样 | 平滑局部响应 |
| `nn.AdaptiveAvgPool2d` | 输出指定空间尺寸 | 常用来适配不同输入大小 |
| `nn.BatchNorm2d` | 按通道归一化激活 | train/eval 行为不同；小 batch 要评估适用性 |
| `nn.Dropout2d` | 按通道丢弃特征图 | 仅用于训练正则化，eval 时关闭 |
| `nn.Flatten` | 展开通道与空间维 | 后接全连接层时核对乘积尺寸 |
| `nn.Linear` | 汇总特征并输出类别分数 | 多分类输出 logits，不提前 softmax |

这些层不是必须全部使用。先建立形状清楚、能训练的基线，再根据数据、验证表现和部署约束调整。

## 15. 常见错误与排查清单

1. **NCHW 与 NHWC 搞混**：打印输入 shape；若数据是 `[N, H, W, C]`，需转换通道顺序，例如 `x = x.permute(0, 3, 1, 2)`。
2. **通道数不匹配**：第一层的 `in_channels` 应等于输入 C；后一层的 `in_channels` 应等于前一层 `out_channels`。
3. **全连接层输入尺寸错误**：逐层打印 shape，或用 `nn.AdaptiveAvgPool2d` 减少对固定高宽的依赖。
4. **空间尺寸算错**：分别代入高度和宽度公式；整数除法按向下取整，池化的 `ceil_mode=True` 是例外设置。
5. **标签与损失不匹配**：`CrossEntropyLoss` 使用类别索引 long 标签；`BCEWithLogitsLoss` 使用浮点的 0/1 目标并与 logits 形状一致。
6. **训练和评估模式混用**：每轮训练前 `model.train()`，验证/推理前 `model.eval()`。
7. **把输出特征图当作概率图**：卷积输出通常是任意实数激活；只有按任务在输出端定义概率变换或使用相应损失后，才有概率解释。
8. **过度下采样**：连续 stride=2 或池化可能很快把小图降到极小尺寸；在每个阶段打印 shape。
9. **仅凭训练准确率判断效果**：同时看验证损失、准确率及类别相关指标；确保数据划分不存在重复样本泄漏。

调试时可以用随机 Tensor 做一次前向传播：

```python
x = torch.randn(2, 3, 64, 64)
for name, layer in model.features.named_children():
    x = layer(x)
    print(name, tuple(x.shape))
```

## 16. 从基础到进阶的延伸方向

学完基础 CNN 后，可以继续学习：

- **经典架构**：LeNet、AlexNet、VGG、ResNet，比较深度、残差连接和下采样设计。
- **残差连接**：让输入与变换后的特征相加，帮助训练更深的网络；相加张量必须具有兼容 shape。
- **转置卷积与上采样**：用于需要增大空间尺寸的生成和分割任务；转置卷积不是严格意义上的卷积逆运算。
- **空洞卷积**：通过 dilation 扩大感受野而不按比例增加参数量。
- **深度可分离卷积**：降低移动端等场景的参数量和计算量。
- **数据增强与迁移学习**：图像任务中常比盲目增大模型更有效；增强应符合标签语义。
- **目标检测与语义分割**：在分类骨干网络之外增加定位、检测或逐像素预测结构。

理解 CNN 时始终追踪四件事：输入输出 shape、每层空间尺寸、通道数变化、某个输出位置覆盖的输入范围。能准确解释这四者，就能读懂大多数基础卷积网络。

## 17. 卷积运算的多通道计算

在深度学习框架中，`Conv2d` 实际执行的是互相关（cross-correlation）：核不先旋转 180 度，而是直接与输入窗口逐项相乘。这与经典信号处理中卷积的定义有差异，但卷积核权重是可学习的，旋转与否不会限制网络可表示的函数；PyTorch API 仍沿用 convolution 的名称。

对输出通道 `o`、空间位置 `(i, j)`，普通卷积的计算可以写成：

$
y_{o,i,j}=b_o+\sum_{c=0}^{C_{in}-1}\sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1}
w_{o,c,u,v}x_{c,\,iS_h+uD_h-P_h,\,jS_w+vD_w-P_w}
$

超出输入范围的位置按 padding 规则处理。也就是说，一个输出通道不是只看输入的一张通道图，而是把它连接到的所有输入通道的局部窗口加权求和，再加一个偏置。

用一个极小例子手算单通道、无 padding、stride=1 的卷积：

```text
输入                 核
1  2  0              1  0
0  1  3              0 -1
2  1  0
```

左上输出为 `1*1 + 2*0 + 0*0 + 1*(-1) = 0`。核向右移动一格，得到 `2*1 + 0*0 + 1*0 + 3*(-1) = -1`。整张 `3 x 3` 输入经过 `2 x 2` 核后输出 `2 x 2`，每个位置都按相同权重计算。

## 18. `1 x 1` 卷积和通道混合

`1 x 1` 卷积的空间窗口只有一个像素，但该位置仍会对所有输入通道做线性组合。因此它可以改变通道数、混合通道信息，而不改变高宽（stride=1 时）：

```python
conv = nn.Conv2d(64, 128, kernel_size=1)
x = torch.randn(4, 64, 20, 20)
y = conv(x)
print(y.shape)  # [4, 128, 20, 20]
```

可把它理解为在每一个空间位置独立应用同一个 `Linear(64, 128)`。它不直接整合相邻像素的空间信息，但常用于瓶颈结构、通道升降维和深度可分离卷积。

## 19. 分组卷积与深度可分离卷积

分组卷积把输入和输出通道划分为若干组，各组之间不连接。`groups` 必须整除 `in_channels` 和 `out_channels`：

```python
group_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=4)
# 每组读取 32/4=8 个输入通道，产生 64/4=16 个输出通道
```

`groups=in_channels` 且 `out_channels` 是 `in_channels` 的整数倍时属于 depthwise convolution；每个输入通道分别做空间卷积（也可为每个输入通道产生多个输出通道）。它本身不跨输入通道混合信息。常见的 depthwise separable convolution 由两步组成：

```python
depthwise = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
pointwise = nn.Conv2d(32, 64, kernel_size=1)
```

第一步提取每个通道内的空间特征，第二步用 `1 x 1` 卷积混合通道。相比直接 `Conv2d(32, 64, 3)`，通常可显著减少权重和乘加运算；实际速度仍取决于设备、内存访问和后端实现，不能只凭参数量推断延迟。

## 20. 池化的窗口、边界与替代设计

池化一般逐通道独立操作：它改变 H/W，不会在不同通道之间求最大值或平均值。对于 `MaxPool2d(kernel_size=2, stride=2)`，输入每个互不重叠的 `2 x 2` 区域被压缩为一个值；若高宽为奇数，最边缘位置如何参与由 padding、stride 和 `ceil_mode` 等设置决定，建议直接打印输出 shape 验证。

最大池化的 padding 区域按负无穷处理，避免补入的零在全负窗口中错误地成为最大值。与卷积不同，池化没有可训练权重；池化输出尺寸通常使用与窗口滑动类似的公式，但 `ceil_mode=True` 会采用向上取整规则。

常见下采样方式包括：

| 方式 | 特性 | 常见使用考虑 |
| --- | --- | --- |
| `MaxPool2d(2)` | 保留局部最大响应 | 简单、常见，但丢弃窗口内其他值 |
| `AvgPool2d(2)` | 汇总局部平均响应 | 更平滑，细小强响应会被稀释 |
| stride=2 的卷积 | 下采样同时学习权重 | 灵活但增加参数和计算 |
| `AdaptiveAvgPool2d((1,1))` | 每通道汇总到固定尺寸 | 常放在分类头之前，不要求特定输入 H/W |

池化不是 CNN 的必需组件。现代网络也常用步长卷积或其他下采样结构；应根据任务是否需要精确空间定位来选择。

## 21. 感受野逐层计算示例

为了把第 9 节的公式落到具体结构，按一维方向追踪等高等宽网络（二维在 H 和 W 上分别应用）：

| 层 | 核/窗口 `k` | 步长 `s` | 本层间隔 `j` | 理论感受野 `r` |
| --- | ---: | ---: | ---: | ---: |
| 输入 | - | - | 1 | 1 |
| Conv 3x3 | 3 | 1 | 1 | 3 |
| MaxPool 2x2 | 2 | 2 | 2 | 4 |
| Conv 3x3 | 3 | 1 | 2 | 8 |

第二层卷积的每个相邻位置在原图上间隔为 2；新增核宽 3 覆盖间隔 `2 * (3-1) = 4`，所以感受野从 4 扩到 8。表中是理论最大感受野，边界位置因 padding 覆盖到的真实像素可能更少；训练得到的有效感受野也可能集中在理论范围的中心。

增大核、堆叠层数、使用步长或 dilation 都会改变感受野，但代价和效果不同。两个 `3 x 3` 卷积叠加可获得 `5 x 5` 的理论范围，同时在两层之间插入非线性；单个 `5 x 5` 卷积则一步覆盖更大范围、参数较多。结构选择还应考虑计算量、优化难度和数据规模。

## 22. 空洞卷积、转置卷积与残差连接

### 22.1 空洞卷积 `dilation`

空洞卷积在核权重之间插入间隔。有效核大小为：

```text
effective_kernel = dilation * (kernel_size - 1) + 1
```

例如 kernel=3、dilation=2 的有效覆盖范围是 5，但仍只有 9 个核权重。它可以在不增加同等规模核参数的情况下扩大感受野。若希望 stride=1 时保持空间尺寸，padding 通常需要根据有效核大小来选，例如该例取 padding=2。过大的 dilation 可能产生稀疏采样带来的栅格效应。

### 22.2 转置卷积 `ConvTranspose2d`

转置卷积常用于生成任务和分割解码器中提高空间分辨率。它对应卷积线性映射的转置结构，不是普通卷积的严格逆操作，也不能保证还原出原输入。输出尺寸还受 `output_padding` 影响；该参数用于消除形状歧义，不是在输出边缘真的补一圈零。若只需放大特征图，也可先用插值（如 `F.interpolate`）上采样，再接普通卷积，二者伪影特征和计算方式不同。

### 22.3 残差连接

残差块把输入 `x` 与卷积变换 `F(x)` 相加：`y = F(x) + x`。相加要求两边 shape 相同；若通道数或空间尺寸变化，需用投影卷积或下采样对齐：

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + residual)
```

恒等捷径给信息和梯度提供直接通路，常用于较深 CNN。残差连接不意味着模型必然更准确，仍需匹配任务、归一化方式和训练配置。

## 23. 图像输入管线：缩放、范围与增强

模型结构正确但输入预处理不一致，效果也会很差。训练和推理必须使用兼容的颜色通道顺序、空间尺寸、数值范围和归一化参数。

若图片像素是 `[0, 255]`，常见转换是转成浮点并缩放到 `[0, 1]`；随后可按训练集统计量逐通道标准化：

```python
# x: float Tensor，形状 [C, H, W]，像素范围 [0, 1]
mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
x = (x - mean) / std
```

这里的数值是一个常见 RGB 预训练模型约定示例，不是所有数据集都应该照搬。若使用预训练权重，应使用该权重对应的 transforms；从头训练时可从训练集计算均值、标准差，并在验证、测试和线上推理中复用训练集的统计量。

常见增强包括随机裁剪、水平翻转、轻微颜色扰动、随机擦除等。增强只应应用于训练集，且必须保持标签语义：例如某些文字识别或左右方向诊断任务不能随意水平翻转。随机增强改变训练样本，每轮看到的图像可能不同；验证和测试变换应确定、可复现。

## 24. 归一化、初始化与 Dropout

### 24.1 BatchNorm 的行为

`BatchNorm2d(C)` 对每个通道的 batch 和空间位置统计均值、方差，再用可学习的缩放和偏移变换。训练时使用当前 batch 统计量并更新 running statistics；评估时通常使用这些运行统计量。因此：

- 训练循环需要 `model.train()`，验证/推理需要 `model.eval()`。
- batch 很小时，统计量噪声可能较大；可考虑 GroupNorm、LayerNorm 或调整 batch/归一化设计。
- `torch.no_grad()` 只关闭梯度，不会自动切换 train/eval 模式。

### 24.2 初始化与激活函数

PyTorch 层会提供默认初始化，建立基线时通常可直接使用。自定义初始化时，应考虑激活函数和网络结构；ReLU 系列常配合 Kaiming 初始化：

```python
nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
if conv.bias is not None:
    nn.init.zeros_(conv.bias)
```

不要在每轮训练中重复初始化参数。初始化只在模型创建阶段进行，且使用预训练权重时不要覆盖已加载参数。

### 24.3 Dropout 与 Dropout2d

`Dropout(p)` 随机屏蔽激活元素；`Dropout2d(p)` 对卷积特征图按通道屏蔽，更适合降低相邻空间激活高度相关时的共适应。二者仅在训练模式生效，评估模式会关闭随机屏蔽。Dropout 是正则化工具，不是修复数据泄漏或验证集设计问题的替代方案。

## 25. 训练质量诊断与实验设计

CNN 项目至少要同时检查训练集和验证集，并保留最终测试集只用于模型选择完成后的评估。训练损失与验证损失的典型解读：

| 训练损失 | 验证损失 | 常见解释 | 优先检查方向 |
| --- | --- | --- | --- |
| 下降 | 下降 | 正常学习 | 继续观察验证表现 |
| 下降 | 先降后升 | 过拟合倾向 | 早停、数据增强、正则化、缩小模型 |
| 高且停滞 | 高且停滞 | 欠拟合或优化未启动 | 检查标签/损失、学习率、模型容量、输入范围 |
| 震荡或发散 | 震荡或发散 | 优化不稳定或数值异常 | 检查学习率、输入尺度、梯度和异常值 |

排查训练“准确率高但验证差”时，依次确认：训练/验证是否来自不同的原始主体或拍摄序列；重复图像、近重复图像是否跨集合；标签映射是否一致；验证阶段是否调用 `eval()`；预处理是否错误地使用全数据统计量；是否反复依据验证集调参造成验证集过拟合。

类别不平衡时，单看 accuracy 可能掩盖少数类失效。至少查看混淆矩阵、precision、recall、F1；二分类还可根据需求观察 ROC-AUC 或 PR-AUC。阈值应由验证集和业务错误代价确定，测试集不能用来反复挑选阈值。

每次实验建议记录随机种子、数据拆分、预处理、模型结构、优化器与学习率、epoch、最佳验证指标、测试指标和权重文件路径。对小型数据集可以进行多随机种子重复或交叉验证，报告均值和波动，而不是只报告一次拆分的结果。

## 26. 用 PyTorch 逐层审计 shape 和参数

调模型时可以注册 hook 自动打印各层输入输出形状：

```python
def print_shape(name):
    def hook(module, inputs, output):
        in_shape = tuple(inputs[0].shape)
        out_shape = tuple(output.shape)
        print(f"{name}: {in_shape} -> {out_shape}")
    return hook

handles = []
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear)):
        handles.append(layer.register_forward_hook(print_shape(name)))

_ = model(torch.randn(2, 1, 28, 28))
for handle in handles:
    handle.remove()
```

统计可训练参数量：

```python
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters: {trainable_parameters:,}")
```

钩子会在每次前向传播时执行，调试完成应移除，避免重复注册导致日志重复或保留不必要的引用。模型可用随机输入通过前向传播，不代表数据管线和训练目标已经正确；还需检查真实 batch、标签和损失。

## 27. CNN 学习自测

尝试不运行代码先回答，再用 Tensor 验证：

1. `[N, 3, 64, 64]` 输入经过 `Conv2d(3, 16, 3, stride=2, padding=1)` 后是什么 shape？
2. 若一层输出 `[N, 32, 14, 14]`，`MaxPool2d(2)` 后是什么 shape？
3. `Conv2d(16, 32, 1)` 会改变哪些维度？它能否直接提取相邻像素的模式？
4. 两层 `3 x 3`、stride=1 的卷积，理论感受野为什么是 5 而不是 6？
5. `model.eval()` 与 `torch.no_grad()` 分别控制什么？为什么验证时通常两者都要用？
6. 为什么测试集不能参与标准化均值计算、早停或阈值挑选？

推荐的答案核验方式是打印逐层 shape、展开卷积输出尺寸公式，并用实际 loss backward 检查梯度是否存在且有限。
