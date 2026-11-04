
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