
(pytorch官方文档)[https://docs.pytorch.org/docs/2.14/torch.html]

(pytorch中文文档)[https://pytorch-cn.readthedocs.io/zh/latest/]

(Learn the Basics)[https://docs.pytorch.org/tutorials/beginner/basics/intro.html]




# PyTorch 深度学习完整学习路线（从 0 到 1）

针对你的目标——成为 AI 大模型应用开发工程师，我建议把 PyTorch 学习分成三个阶段：

1. 掌握 `torch` 基础 API：能够操作张量、构建神经网络、完成模型训练。
2. 掌握深度学习核心原理：理解反向传播、优化器、CNN、RNN、Attention、Transformer。
3. 面向大模型开发：理解 Embedding、Transformer、预训练、微调与推理。

这条路线以官方 PyTorch 的基础教程为起点，再逐步过渡到深度学习模型和大模型。官方基础教程覆盖 Tensor、数据加载、构建模型、自动求导、优化循环以及模型保存与加载。


cdn-agent、cdn-gateway、cdn-scheduler、pcdn-agent、protobuf


## 一、整体路线图

### 阶段 1：PyTorch 基础

Tensor · autograd · nn · optim · DataLoader

### 阶段 2：深度学习训练原理

前向传播 · 反向传播 · 损失函数 · 优化算法

### 阶段 3：经典神经网络

MLP · CNN · RNN · LSTM · Attention

### 阶段 4：Transformer 与大模型

Embedding · Self-Attention · Transformer · GPT

### 阶段 5：大模型训练与应用衔接

Hugging Face · LoRA · 量化 · 推理 · 部署

这里有一个重要区别：学习 `torch` 包，不等于学习全部深度学习，也不等于学习全部大模型应用开发。 你需要先把框架用熟，再用它理解神经网络，最后把知识迁移到 Transformer 和大模型。

## 二、阶段 1：系统学习 `torch` 包基础

目标：能够独立编写一个完整的深度学习训练程序。

建议按以下顺序学习，而不是按 Python 包的目录顺序逐个看 API。

现在我已经学习了torch包下的 tensor、设备与张量、自动求导、torch.nn模块、优化器、损失函数、数据集与批量加载以及懂得了一个完整的训练、验证、推流流程，接下来我还想更进一步的学习深度学习，那么接下来我的学习路线是什么？

### 第 1 章：Tensor 张量操作

这是整个路线的起点。

| 学习内容  | 重点 API                                                                   |
| ----- | ------------------------------------------------------------------------ |
| 张量创建  | `tensor()`、`zeros()`、`ones()`、`randn()`                                  |
| 张量属性  | `shape`、`ndim`、`dtype`、`device`                                          |
| 维度操作  | `view()`、`reshape()`、`squeeze()`、`unsqueeze()`、`permute()`、`transpose()` |
| 张量计算  | `sum()`、`mean()`、`max()`、`matmul()`                                      |
| 张量组合  | `cat()`、`stack()`、`split()`、`chunk()`                                    |
| 索引与掩码 | 切片、布尔索引、`masked_fill()`                                                  |
| 数据转换  | `.to()`、`.float()`、`.long()`、`.item()`                                   |

特别注意：张量维度、广播机制、矩阵乘法、数据类型。这些知识会直接影响你以后理解 Attention 中的 \\(Q,K,V\\) 运算。

练习项目： 不依赖 NumPy，实现一个简单的批量矩阵运算程序，自己追踪每一步的张量形状。

### 第 2 章：设备与张量内存

重点学习：

- CPU、CUDA、MPS 的区别。
- `torch.device()`、`.to(device)`。
- `torch.cuda.is_available()`、`torch.backends.mps.is_available()`。
- `float32`、`float16`、`bfloat16`。
- 张量复制、共享内存、`clone()`、`detach()`。
- 连续内存与 `contiguous()`。

练习项目： 在你的 Apple Silicon Mac 上运行同一个张量计算程序，检查张量实际所在的设备，并理解为什么模型和输入需要位于兼容的设备上。

### 第 3 章：自动求导 `torch.autograd`

这是最值得花时间理解的一章。

| 学习内容               | 必须理解的内容               |
| ------------------ | --------------------- |
| `requires_grad`    | 哪些张量需要梯度              |
| `grad_fn`          | 运算如何连接计算图             |
| `backward()`       | 如何计算梯度                |
| `.grad`            | 梯度保存在哪里               |
| 梯度累积               | 为什么需要清零梯度             |
| `detach()`         | 如何切断梯度传播              |
| `no_grad()`        | 如何暂时关闭梯度计算            |
| `inference_mode()` | 推理场景下如何减少 autograd 开销 |

练习顺序：

1. 手动计算 \\(y=3x^2+2x+1\\) 的导数。
2. 使用 `backward()` 自动求导。
3. 验证自动计算结果是否与手工计算一致。
4. 观察多个 `backward()` 调用时梯度如何累积。
5. 用梯度下降拟合 \\(y=3x+2\\)。

你之前已经做过简单线性回归，建议这次重点理解梯度的计算过程，以及梯度如何影响参数更新。

### 第 4 章：`torch.nn` 神经网络模块

建议按照“单层 → 多层 → 常用网络组件”的顺序学习。

| 内容   | 必学 API                                             |
| ---- | -------------------------------------------------- |
| 网络结构 | `nn.Module`、`nn.Sequential`、`forward()`            |
| 全连接层 | `nn.Linear`                                        |
| 激活函数 | `nn.ReLU`、`nn.GELU`、`nn.Sigmoid`、`nn.Softmax`      |
| 正则化  | `nn.Dropout`                                       |
| 归一化  | `nn.BatchNorm1d`、`nn.LayerNorm`                    |
| 嵌入层  | `nn.Embedding`                                     |
| 参数管理 | `parameters()`、`named_parameters()`、`state_dict()` |
| 模式切换 | `train()`、`eval()`                                 |

其中 `nn.Embedding` 和 `nn.LayerNorm` 对以后理解 Transformer 尤其重要。

练习项目： 从零构建一个多层感知机 MLP，完成 MNIST 手写数字分类，并观察隐藏层维度、激活函数对结果的影响。

### 第 5 章：损失函数与优化器

这里要将 `nn` 与 `optim` 联系起来学习。

损失函数：

- `nn.MSELoss`
- `nn.L1Loss`
- `nn.BCEWithLogitsLoss`
- `nn.CrossEntropyLoss`

优化器：

- `torch.optim.SGD`
- `torch.optim.Adam`
- `torch.optim.AdamW`

进阶内容：

- 学习率 `lr`
- 权重衰减 `weight_decay`
- 动量 `momentum`
- `torch.optim.lr_scheduler`
- 梯度裁剪 `clip_grad_norm_`

需要特别理解：`CrossEntropyLoss` 通常直接接收模型输出的 logits，而不是提前做过 Softmax 的概率。二分类中也要区分 `BCEWithLogitsLoss` 与 `BCELoss`。

练习项目： 对同一个分类数据集分别使用 SGD、Adam 和 AdamW，比较训练损失和验证集指标。

### 第 6 章：数据集与批量加载

重点 API：

- `torch.utils.data.Dataset`
- `torch.utils.data.DataLoader`
- `TensorDataset`
- `Subset`
- `random_split`
- `torchvision.transforms`

必须理解：

- `Dataset` 如何定义单条样本。
- `DataLoader` 如何组成 batch。
- `batch_size`、`shuffle`、`num_workers` 的作用。
- 训练集、验证集、测试集如何划分。
- 为什么数据预处理要避免数据泄漏。

练习项目： 将一个 CSV 数据集转换成 Tensor，编写自定义 `Dataset`，再用 `DataLoader` 完成训练。

### 第 7 章：完整训练、验证、推理流程

把前面所有知识串联起来，形成稳定的训练模板：

```python
for epoch in range(epochs):    
    model.train()    
    for X, y in train_loader:
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()    
        model.eval()    
    with torch.inference_mode():        
        for X, y in valid_loader:            
            pred = model(X)  # 计算验证集指标
```

还要学习模型保存与加载、检查点、训练中断恢复、损失曲线、过拟合判断。

阶段验收： 不照抄教程，独立完成一个分类或回归项目，包括数据处理、模型定义、训练、验证、指标评估和模型保存。

## 三、阶段 2：深入理解深度学习

基础 API 学会之后，不要继续无限扩展 API 清单。应开始学习不同神经网络的结构、原理和适用场景。

[#machinelearning #datascience #deeplearning #neuralnetwork | AMIT SHEKHAR](https://images.openai.com/static-rsc-4/-USVUkwieLffJPrajtAQycvZuveMAWwxjiFsQQG233S4SnZ_4sWJddbbI9cz7dOwKTHYMLxR7XGfqMxRnGW1WX5E4OLoCVf-wubuOk08-_WLXdoTLAPlsct0iccE2GdjiDLvxVnifNGI43rHwPXF0msTRhoTTcL8R5WBucDkxVU?purpose=inline)

第 8 章：MLP 多层感知机

学习前向传播、反向传播、激活函数、梯度消失与梯度爆炸、初始化、正则化。

项目：房价预测、分类任务、训练曲线分析。

[AcaDraw - Outil IA pour dessins scientifiques | Illustrations en quelques secondes](https://images.openai.com/static-rsc-4/DKs-srh42CuSYoBKJCOkuegFPmjI13YHdv5IBf02BC5fq9sqaCfRIgxLGAOSG_JZPrGtjv5S-gsxJ9KKYthKyVMliQaV_1QXiJbzx2pqgL18YQSec4C43LaIcjPXO2qW3S4bbpY-MswedZ3ZbEpPMINUL3WORjAZxaFdtAibDaI?purpose=inline)

第 9 章：CNN 卷积神经网络

学习 `nn.Conv2d`、`nn.MaxPool2d`、卷积核、步长、填充、特征图、感受野。
    
项目：CIFAR-10 图像分类。

[LSTM - Jing's blog](https://images.openai.com/static-rsc-4/AwsqfITnd8KvjC90SfXM83MgI8K3NTN3Pu3dC9H5e62oXnbsc07B3s8AYsZvgvLc208TUdeFA8y6i9nGrNoMuswI-8Pl8g9yn1qpPs-4q2fdHC3bd0OB4SlM3_AsD_RB1wA_oeumHEqtjaEUEzMDRKmmJLEw-_TSG7AQ2qWTwZA?purpose=inline)

第 10 章：RNN 与 LSTM

学习序列建模、隐藏状态、时间步、梯度消失、`nn.RNN`、`nn.LSTM`、`nn.GRU`。

项目：时间序列预测或文本分类。对大模型目标而言，这一章可以适当简化。

[一文看懂Transformer到BERT模型-CSDN博客](https://images.openai.com/static-rsc-4/0K3tYZ7a0gMbOVbNdj4heoPort0lClaGsyCPLgg0Ijbs5O_zYWdBrOgn6cgOgT_P1tvdRnUqIoAbV3R_NjoJzA003blE42ADHmr1uXQifL19VkQWk6a8zr1UI5jy9VaOrtzBnOPBeDQB0ArbUrdsUbut4FF5KTDhUHemu9_Nr3s?purpose=inline)

第 11 章：Attention 注意力机制

学习 Query、Key、Value，矩阵乘法、缩放点积、Softmax、注意力掩码、多头注意力。

项目：使用 PyTorch 手写 Scaled Dot-Product Attention。

对于你的目标，MLP 和反向传播需要扎实掌握；CNN、RNN 学到能理解网络结构即可；Attention 和 Transformer 则需要深入学习。

## 四、阶段 3：专门学习 Transformer 与大模型

这是从 PyTorch 深度学习走向大模型应用开发的关键阶段。

| 顺序 | 学习内容                 | 必须理解的重点                         |
| -- | -------------------- | ------------------------------- |
| 12 | Embedding            | Token ID 如何转换成向量                |
| 13 | 位置编码                 | 正弦位置编码、RoPE 的基本思想               |
| 14 | Self-Attention       | Q、K、V、注意力权重、掩码                  |
| 15 | Multi-Head Attention | 多头拆分、拼接、投影                      |
| 16 | Transformer Block    | 残差连接、LayerNorm、FFN、GELU/SwiGLU  |
| 17 | GPT 结构               | Decoder-only、因果注意力、下一个 Token 预测 |
| 18 | 训练与微调                | Tokenization、交叉熵、预训练、SFT、LoRA   |
| 19 | 模型推理                 | KV Cache、生成策略、量化、显存管理           |

建议亲手使用 PyTorch 实现一个迷你 GPT，而不是一开始就尝试从头训练大型模型。

先完成字符级语言模型，再理解 Tokenizer、预训练模型和 Hugging Face Transformers。PyTorch 官方也提供了 Transformer 组件与性能优化教程，可在基础扎实后继续深入。

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

PyTorch Tutorials 2.14.0+cu130 documentation

+1



## 五、阶段 4：工程化和性能优化

这部分并非一开始的重点，但做模型微调和推理时会用到。

| 模块    | 学习内容                    |
| ----- | ----------------------- |
| 混合精度  | `torch.amp`、FP16、BF16   |
| 显存优化  | 梯度累积、梯度检查点、激活内存         |
| 性能分析  | `torch.profiler`        |
| 编译加速  | `torch.compile`         |
| 分布式训练 | `torch.distributed`、DDP |
| 模型部署  | 推理模式、批量推理、模型导出          |

`torch.compile` 可以优化模型执行，但编译过程有额外开销，因此建议先掌握正确的训练与推理流程，再研究它。

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

PyTorch Tutorials 2.14.0+cu130 documentation

+1



对于大模型应用开发，分布式训练、底层 GPU 优化不需要立即深入。优先掌握模型加载、推理、微调和显存基础更实际。

## 六、建议的项目顺序

## 项目实战清单

0/6 已完成

1\. 线性回归

理解 Tensor、梯度与优化器

验收：独立拟合 y = 3x + 2

2\. MNIST 分类

掌握 nn.Module 与训练循环

验收：训练、验证、保存和加载模型

3\. CIFAR-10

理解 CNN 与特征提取

验收：比较 MLP 与 CNN

4\. 文本分类

掌握 Embedding 与序列数据

验收：实现简单文本分类模型

5\. 迷你 GPT

理解 Transformer 的核心结构

验收：完成自回归下一个 Token 预测

6\. 开源模型微调

衔接大模型开发

验收：使用预训练模型完成小规模 LoRA 微调

## 七、学习时间安排

如果每天学习 1～2 小时，可以先按这个节奏规划，实际进度以项目完成情况为准。

| 周次      | 学习任务                     |
| ------- | ------------------------ |
| 第 1 周   | Tensor、形状变换、设备、数据类型      |
| 第 2 周   | autograd、梯度下降、MLP        |
| 第 3 周   | 损失函数、优化器、DataLoader      |
| 第 4 周   | 完整训练流程、MNIST 项目          |
| 第 5 周   | CNN、RNN/LSTM 基础          |
| 第 6 周   | Embedding、Attention      |
| 第 7～8 周 | Transformer、迷你 GPT       |
| 后续      | Hugging Face、LoRA、推理与工程化 |

这是一份初步计划，不代表八周就能精通深度学习；如果反向传播和张量运算还不熟，可以给基础阶段多安排时间。

## 八、你现在最应该从哪里开始？

我建议你先完成第一阶段的基础训练，再进入 CNN 和 Transformer。不要一开始就同时学习 PyTorch、LangChain、LangGraph 和大模型微调，否则容易只会调用接口，却不理解模型内部的数据流。

官方教程可以从这里开始：

- [PyTorch 基础教程（官方）](https://docs.pytorch.org/tutorials/beginner/basics/intro.html?utm_source=chatgpt.com)
- [PyTorch 教程总目录](https://docs.pytorch.org/tutorials/?utm_source=chatgpt.com)
- [Transformer 组件与性能优化（官方）](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html?utm_source=chatgpt.com)

如果你希望我继续带你按章节学习，可以补充你偏好的讲解方式和目前的掌握程度，我会据此细化每章的 API、原理和项目练习。

细化学习路线









Deep Learning（深度学习）除了 CNN、RNN，还有很多经典且值得掌握的模型。不过，并不是所有模型都需要深入学习。对于你希望走的 AI 大模型应用开发工程师方向，我建议按照「基础网络 → 经典架构 → Transformer → 大模型」这条路线学习。

先给你一个整体结论：

必须重点掌握的模型家族：MLP、CNN、RNN/LSTM/GRU、Autoencoder、GAN、Attention、Transformer。

其中，Transformer 是连接传统深度学习与现代大语言模型的核心。
一、Deep Learning 经典模型全景图
Deep Learning
深度学习经典模型体系
基础神经网络

MLP / Feed Forward Network

CNN 家族

LeNet、AlexNet、VGG、ResNet、U-Net


RNN 家族

Simple RNN、LSTM、GRU、BiRNN


生成模型

Autoencoder、VAE、GAN、Diffusion


注意力机制

Attention、Self-Attention、Cross-Attention



Transformer 家族

Encoder、Decoder、Encoder-Decoder

BERT、GPT、T5、ViT

现代大模型体系

LLM、多模态模型、扩散模型、MoE

这是模型家族之间的学习关系示意，不代表所有模型严格按此顺序演进。
二、除了 CNN、RNN，还必须学习哪些经典模型？
1. MLP（多层感知机）

必须掌握
 基础核心









machinelearninggeek.com


MLP（Multilayer Perceptron）是最基础的前馈神经网络。

核心结构：
输入层
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
输出层


核心知识：
- 全连接层 nn.Linear
- 激活函数 ReLU、Sigmoid、Tanh、GELU
- 前向传播与反向传播
- 损失函数与优化器
- Dropout、BatchNorm
- 初始化与梯度问题
经典应用：
- 房价预测
- 用户流失预测
- 手写数字分类
- 表格数据分类与回归
你之前学习过的 784 → 256 → 128 → 10 MNIST 模型，本质上就是 MLP。
2. LSTM、GRU、双向 RNN

必须掌握
 序列模型



































它们都属于 RNN 家族，但解决的问题和结构有所不同。
模型	核心特点	典型应用
Simple RNN	基础循环结构	简单序列预测
LSTM	使用门控和细胞状态	长序列建模
GRU	简化门控结构	时间序列、文本
BiRNN	同时利用前向和反向信息	文本分类、序列标注

重点掌握：
- RNN 的梯度消失与梯度爆炸
- LSTM 的输入门、遗忘门、输出门
- GRU 的更新门、重置门
- 双向网络的输出维度
- Padding、Embedding、PackedSequence
你已经接触过 RNN、LSTM 和 Embedding，接下来把 GRU、双向 LSTM 补上即可。
3. Autoencoder（自编码器）

推荐掌握
 无监督学习









ibm.com


Autoencoder 的核心思想是：

将数据压缩成低维特征，再通过解码器重建原始数据。

结构：
原始输入
   ↓
Encoder（编码器）
   ↓
Latent Space（潜在空间）
   ↓
Decoder（解码器）
   ↓
重建数据


例如：
784 维图像
   ↓
256 维
   ↓
64 维
   ↓
16 维（压缩特征）
   ↓
64 维
   ↓
256 维
   ↓
784 维（重建图像）


核心知识：
- Encoder / Decoder
- Latent Representation
- Reconstruction Loss
- MSELoss
- 降维与特征提取
经典应用：
- 图像降噪
- 异常检测
- 数据压缩
- 特征学习
推荐项目：使用 Autoencoder 对 MNIST 手写数字进行降噪和重建。
4. VAE（变分自编码器）

推荐掌握
 生成模型









paloaltonetworks.com


VAE（Variational Autoencoder）可以看作 Autoencoder 的概率生成模型扩展。

普通 Autoencoder：
输入 → Encoder → 固定潜在向量 → Decoder → 重建数据


VAE：
输入
 ↓
Encoder
 ↓
均值 μ、方差参数 log σ²
 ↓
从概率分布中采样 z
 ↓
Decoder
 ↓
生成数据


核心知识：
- 概率分布
- 均值与方差
- Reparameterization Trick（重参数化技巧）
- KL Divergence
- Reconstruction Loss
- ELBO
经典应用：
- 图像生成
- 潜在空间表示学习
- 数据生成
VAE 是理解生成模型的重要基础，但如果你的目标是 LLM 应用开发，可以先了解原理，不必一开始就深入推导所有数学细节。
5. GAN（生成对抗网络）

推荐掌握
 生成模型









ijpds.org


GAN（Generative Adversarial Network）由两个网络组成：
- Generator（生成器）：负责生成假数据。
- Discriminator（判别器）：判断输入数据是真实数据还是生成数据。
训练过程：
随机噪声
   ↓
Generator
   ↓
生成假图片
   ↓
Discriminator ← 真实图片
   ↓
判断真假
   ↓
两个网络对抗训练


经典模型：
模型	特点
GAN	基础生成对抗网络
DCGAN	使用 CNN 生成图像
Conditional GAN	根据条件生成数据
CycleGAN	图像风格转换
StyleGAN	高质量人脸生成

重点理解 Generator、Discriminator、对抗损失和训练稳定性。
6. Attention（注意力机制）

必须掌握
 LLM 核心基础









aignishant.medium.com


Attention 是现代 Transformer 和大语言模型的核心机制之一。

它的核心思想是：

让模型根据当前任务需要，动态计算不同信息的重要程度。

核心公式：
\[
Attention(Q,K,V)=
Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

其中：
- Q（Query）：查询向量。
- K（Key）：键向量。
- V（Value）：值向量。
- \(QK^T\)：计算不同位置之间的相关性。
- Softmax：把相关性转换成权重。
- 最后与 V 相乘，得到加权信息。
必须学习：
1. Scaled Dot-Product Attention
2. Self-Attention
3. Multi-Head Attention
4. Cross-Attention
5. Masked Attention
Attention 本身不是一个独立的完整网络架构，而是一种可以嵌入网络中的机制。
7. Transformer

最高优先级
 大模型核心









medium.com


Transformer 是你从传统深度学习转向大模型开发必须掌握的架构。

它不依赖传统 RNN 的逐时间步循环计算，而是通过 Attention 建模序列中不同位置之间的关系。

经典结构：
输入 Token
   ↓
Embedding
   ↓
Positional Encoding
   ↓
Multi-Head Attention
   ↓
Feed Forward Network
   ↓
Add & Norm
   ↓
输出


需要重点学习三个架构类型：
类型	代表模型	典型应用
Encoder-only	BERT	文本理解、分类
Decoder-only	GPT	文本生成、大语言模型
Encoder-Decoder	T5	文本转换、翻译

经典模型：
- Transformer
- BERT
- GPT
- T5
- Vision Transformer (ViT)
对你而言，Transformer 应当是传统深度学习学习阶段的重点收尾内容。
8. Diffusion Model（扩散模型）

推荐掌握
 现代生成模型









itismevarnica.medium.com


扩散模型是现代图像生成的重要技术路线。

核心思想：
- 正向过程：逐渐向真实数据添加噪声。
- 反向过程：训练模型逐步去除噪声，恢复数据。
真实图片
   ↓
添加噪声
   ↓
更多噪声
   ↓
纯噪声

反向生成：

纯噪声
   ↓
逐步去噪
   ↓
逐步恢复结构
   ↓
生成图片


经典代表：
- DDPM
- DDIM
- Latent Diffusion
- Stable Diffusion
如果未来想做多模态、图像生成、视觉大模型，建议深入学习。
9. GNN（图神经网络）

了解并按需深入
 图数据









upgrad.com


GNN（Graph Neural Network）主要处理图结构数据。

例如：
用户 A ─── 用户 B
  │          │
  │          │
用户 C ─── 用户 D


普通 CNN 处理规则网格数据，RNN 处理序列数据，而 GNN 主要处理节点、边和它们之间的关系。

经典模型：
- GCN
- GraphSAGE
- GAT
- GIN
应用：
- 社交网络分析
- 推荐系统
- 知识图谱
- 分子结构预测
如果未来涉及知识图谱、推荐系统或图数据检索，再深入学习。

三、哪些模型最值得优先学习？
考虑到你的目标是 AI 大模型应用开发工程师，我把它们分成三个级别。
优先级	模型	学习要求
P0	MLP	熟练掌握
P0	CNN	熟练掌握
P0	RNN / LSTM / GRU	理解并能实现
P0	Attention	深入掌握
P0	Transformer	深入掌握
P1	Autoencoder / VAE	理解并能实现基础项目
P1	GAN	理解核心原理
P1	Diffusion	理解核心原理
按需	GNN	根据业务方向学习

四、推荐的完整学习路线
我建议你按照下面的顺序学习，而不是按模型出现的历史顺序逐个学习。
PyTorch 深度学习学习路线
1. 阶段 1：基础神经网络
   - MLP
   - 激活函数
   - 损失函数
   - 优化器
   - BatchNorm / Dropout
     重点掌握
2. 阶段 2：计算机视觉
   - CNN
   - LeNet
   - AlexNet
   - VGG
   - ResNet
   - U-Net
     重点掌握
3. 阶段 3：序列模型
   - Simple RNN
   - LSTM
   - GRU
   - Bidirectional RNN
   - Embedding / Padding
     重点掌握
4. 阶段 4：生成模型
   - Autoencoder
   - VAE
   - GAN
   - Diffusion
     理解原理，完成基础项目
5. 阶段 5：Attention
   - Attention
   - Self-Attention
   - Multi-Head Attention
   - Cross-Attention
   - Mask
     深入掌握
6. 阶段 6：Transformer
   - Transformer Encoder
   - Transformer Decoder
   - BERT
   - GPT
   - T5
   - ViT
     深入掌握
7. 阶段 7：大模型基础
   - Tokenizer
   - Pretraining
   - SFT
   - LoRA / QLoRA
   - RLHF / DPO
   - RAG
   - Agent
     进入目标方向

五、每个阶段建议完成的 PyTorch 项目
模型	推荐实战项目
MLP	MNIST 手写数字分类
CNN	CIFAR-10 图像分类
ResNet	CIFAR-10 ResNet 残差网络
LSTM / GRU	IMDb 文本分类、股票预测
Autoencoder	MNIST 图像降噪
VAE	MNIST 数字生成
GAN	MNIST 手写数字生成
Attention	手写 Self-Attention
Transformer	Transformer 英德机器翻译
BERT	IMDb 文本情感分类
GPT	字符级 GPT 文本生成
Diffusion	MNIST 简单图像生成

我的建议：你目前已经接触 MLP、CNN、RNN、LSTM、Embedding 和 Padding，不必回头重复学习。下一步先把 GRU 和经典 CNN 架构补齐，然后集中精力学习 Attention 和 Transformer。 这两部分对你后续学习 GPT、LangChain、LangGraph 和大模型应用开发最有价值。

如果继续推进，我可以下一步给你制定一份Transformer 从零到一的 PyTorch 学习路线，包含每个知识点、数学公式、代码练习和完整项目。
🧠明天开始学习 Transformer 路线轻点设置


torch11_Autoencoder
torch12_NLP
torch13_Attention
torch14_Transformer 
torch15_BERT
torch16_GPT
torch17_GNN


