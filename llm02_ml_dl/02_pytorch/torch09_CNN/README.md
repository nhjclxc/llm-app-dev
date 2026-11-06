CNN（卷积神经网络）除了 CIFAR-10 图像分类，还有很多经典项目。

如果你正在系统学习 PyTorch 深度学习，我建议不要只做分类项目，而是按照 图像分类 → 目标检测 → 图像分割 → 人脸识别 → 图像生成 逐步学习。

这几个方向能够覆盖 CNN 最核心的应用场景。

# 一、CNN 经典项目路线

我把经典项目分成 6 个方向，按照学习价值和难度排列。

## 1. 图像分类（Image Classification）



建议优先掌握

入门



任务：输入一张图片，判断图片属于哪个类别。

[Introduction to Deep Learning](https://images.openai.com/static-rsc-4/8UHiz7yKTjL6E0H1KGfWlYrgrkOQ2P1Fd4oBcCzUjcWG3_pA9c_k3OZ3gjLYCIH_8t291Nr4GLEx8JT1y6G0-bmrLRGOfmfUHAmhhCmFzPhYUFRBhLJnMOGl9Pnbp4vFwIXmK9nmZ7k693vZCfcYqN-8SsNXFk4aqoMfnrrxFr8?purpose=inline)

项目 1：MNIST 手写数字识别

- 数据集：MNIST
- 类别：0～9
- 模型：LeNet-5
- 目标：识别手写数字

[MNIST Fashion Classification | Matt Pewsey](https://images.openai.com/static-rsc-4/f3T6eWv8lXNG2kYoVwaH4JdxQc3R_qg9Pv94kUyPwXspLLFS8mP4nNTJ4DXV0K-Xp0q5mKCntMn4M2Y3fkoJg-wCTyVP5c4fSowsaZYI1XoCw18pxja_nj3egsW0zGjtvoz6erfgvGBxTVC-4FeLO15Hlzv0YCBSwC9hSC1eBRg?purpose=inline)

项目 2：服装图像分类

- 数据集：Fashion-MNIST
- 类别：10 种服装
- 模型：CNN
- 目标：识别衣服类型

[General | Vision Datasets](https://images.openai.com/static-rsc-4/zGSTZUmAm7A_9O1Q-SDodQkm5fsB58S-FX33DSrV7Lmmp3W04tNZsuuWocPdqAjFj8TgfkwCV1K7ZeC4RM7U1ZI2BDPcIiVyDAO8DHc7HeymtLjxJYIeMgMaNTAf1z7yUCBcNOALNxJD3Oz1n2AHv4iSUjo1WSPCXffeRUbVItg?purpose=inline)

项目 3：自然场景分类

- 数据集：Intel Image Classification
- 类别：建筑、森林、冰川等
- 模型：VGG、ResNet
- 目标：识别自然场景

[Transfer learning and fine-tuning  |  TensorFlow Core](https://images.openai.com/static-rsc-4/o_YLbOIPZ98Q2RtGwPPiqYrk11az-ghn20QrwAVa_l5ploKnc6ufVZ0eDW087-4fVOnKitJRCdYERnnxsaMNhqmFlBiXbP7DTjxhZaghiJwuYnkPTPXJakphXN_gptb4bk_ZiPTYlPJZ6daTNjVZKM67c9rGG8jZsYH_AfYy9yM?purpose=inline)

项目 4：猫狗分类

- 数据集：Cats vs Dogs
- 类别：猫、狗
- 模型：CNN、ResNet
- 目标：判断图片中的动物

这个阶段重点学习：

- CNN 卷积、池化、感受野。
- BatchNorm、Dropout。
- LeNet、AlexNet、VGG、ResNet。
- 数据增强、过拟合、迁移学习。
- 混淆矩阵、Precision、Recall、F1。

## 2. 目标检测（Object Detection）

中级

任务：不仅识别图片中的物体，还要找出物体在图片中的位置。

[YOLOv3 Object Detection in TensorFlow 2.x | by Anushka Dhiman | Analytics Vidhya | Medium](https://images.openai.com/static-rsc-4/Gy8l9ggQZHIOHCJMCmzBpzDVmsxU0UfC5MRp5RBtRG66bqBlBrR33kOS99Ywvkror3vr-P8GtYTjdd0zLq2kITg0YSdHTWnXJSEP1csyCK8Pz_BVev00KF-M1XilloSTJp5jE84pZLl0skBCcZ0H2xyQ8y-3Wl7ME09Gt2RCrLE?purpose=inline)

[目标检测数据集PASCAL VOC简介 | arleyzhang](https://images.openai.com/static-rsc-4/9PUfObhDII9LRIgLEac-BqhhLh4MKQmGX2LrJ_nMDcD7IuoAWtwc1-btNLluMc8V-Kfdv_aUEMN5kj5nV4_ozzPCz3obuMZBjxh2CmYgYt5jLnVwMrThAAqyOSCPbpYN1PGwqR9ZH1jov8Yx8ZDcv6TD5kWlhOc5g05ZpBiCSQA?purpose=inline)

[Traffic-sign detection | ViCoS Lab](https://images.openai.com/static-rsc-4/sTy__P624bR-ogpI9IN5sIWKnz2iDRsXK859nrp_M-DjipoqZWJKiqxnMkI5Rf8GNdyXKtb6NalPK9mzMnv8HZEJY59kNeqGKElBWvru20_RetPvTcbaOPLlIh-cH6ztklULC0ZG7mHYSE5ggnS6lkxIMoFsPelGzSORqpU-jE8?purpose=inline)

8

| 经典项目   | 数据集                   | 主要模型              | 学习目标        |
| ------ | --------------------- | ----------------- | ----------- |
| 交通标志检测 | GTSDB                 | YOLO              | 识别并定位交通标志   |
| 行人检测   | Penn-Fudan Pedestrian | Faster R-CNN      | 检测行人位置      |
| 多目标检测  | Pascal VOC            | SSD、YOLO          | 同时检测多个物体    |
| 通用物体检测 | COCO                  | YOLO、Faster R-CNN | 检测 80 类常见物体 |

重点学习：

- Bounding Box（边界框）。
- IoU（交并比）。
- NMS（非极大值抑制）。
- Anchor Box。
- Precision、Recall、mAP。
- YOLO、SSD、Faster R-CNN。

推荐项目：YOLO 交通标志检测。

它能帮助你理解 CNN 如何从单纯分类发展到目标定位和检测。

## 3. 图像分割（Image Segmentation）

中高级

任务：对图片中的每一个像素进行分类，而不仅仅是识别整张图片。

[TensorFlow U-Net Melanoma Segmentation (ISIC 2018) | Keras Tutorial | by Eran Feit | Image Segmentation tutorials | Medium](https://images.openai.com/static-rsc-4/Ys8rHBzy3OmKJmU5FXXc74oMgI9TWZwim6YbptUr0Ugg4Jj__4iJstE4QnNW6FHT-kJg_Y5neX3VLGx1JNUeV09_77jcKvuOq9nStOZs3w2Xjja2rBg0M51zF0jNZiIurt3maSc6NkTCytxorxLeteqXX5eBzh_DsyPWVllOFdE?purpose=inline)

[Examples – Cityscapes Dataset](https://images.openai.com/static-rsc-4/f7n9UOOU_Gs9eABprXyQKdu93oGWB8bPeIa1PhuBvGryOGR1Dcy8Uu3hIoOHhj2CVJjpYDL2pHHV5htBO_dtpVT6GOpqL4TF5aXqL9jyclkv_Bc9IQYpMkX-Va0lG4qFj_H224VmrKnmQhaol5oKqVQ9O8Jgn49fu1PcgJq3urA?purpose=inline)

[Mask RCNN －原理解析（一）\_mask rcnn原理 to be better one-CSDN博客](https://images.openai.com/static-rsc-4/Hh9Z4o864RPLcj5B0UHhWwsyL316Y0TxANVQBsCeSouGbFNFl9CVGTDxWwYsWrUWuHL-6W3kpbDfYv8P3_YXpiCxeMiEUw9K8LFWoU9tEZ_yPhBtjz7hnZtEOhtaQ-lcvByb4QOB3QWJMQ5P1H6vEz5NvqdrMyREt8bUKFXhE6o?purpose=inline)

9

| 经典项目   | 数据集                               | 主要模型               | 学习目标       |
| ------ | --------------------------------- | ------------------ | ---------- |
| 医学细胞分割 | Cell Segmentation                 | U-Net              | 分割细胞区域     |
| 道路场景分割 | Cityscapes                        | FCN、DeepLab        | 分割道路、车辆、行人 |
| 肺部图像分割 | Chest X-ray segmentation datasets | U-Net              | 分割肺部区域     |
| 人像分割   | COCO                              | DeepLab、Mask R-CNN | 分割人物区域     |

重点学习：

- FCN（全卷积网络）。
- U-Net。
- Encoder / Decoder。
- Transposed Convolution。
- Dice Loss。
- IoU、mIoU。

推荐项目：U-Net 医学图像分割。

这是非常经典的 CNN 进阶项目，能够帮助你理解卷积网络如何保留空间信息。

## 4. 人脸识别与人脸相关任务

中高级

[Medium](https://images.openai.com/static-rsc-4/l0CxWxmgz-0vAByO8IubylFnruwwZVQgt2PD_tkLtnOEOApci_CuxyX8vGaqS7S_s9hkYVjz0K_C4ChK6yTT1EJIdZMxpcG7g6Es2TPHWlrXjozJEPzPXsIKTQHMaUuZVF-xFrRAens3SiKE2zatNM7OZ80QgV_Dcvs-Z1BQZYI?purpose=inline)

[Guide de détection de visages  |  Google AI Edge  |  Google AI for Developers](https://images.openai.com/static-rsc-4/5QraghMBASm-IMa9Tm4S6U8zGv6n9mjd_Au-NvRpi8Egm_Pv2xuKZ-WmI--XOKulNjR3AxHPSndrAjsO0iOO4C-dKW4-pUx8wWUZlBf3kxajg8WQxyhVUWeJdfzXmRe4wOM-o5HcTdZ4xBdsYnfavBsNG217OyeBO3Qje6npbYs?purpose=inline)

[Facial Landmark Detection Algorithms | by Raji Lini | CodeX | Medium](https://images.openai.com/static-rsc-4/Qpr36t0B3ok1U3nmsujYz_H-rQ2bOI130vImrB1m9I5x_HZB3bSsNw2SKBa2lxTD0jz_SowEtNeu8ZTVxxw7x5ijWbPid0xPBsGNY3I0t4XUMqrjoAN60oD9gG-xYr2bvENWnAtqZVWctourzuFKOybOHr4C9Kby3J1FUqkc65g?purpose=inline)

8

| 经典项目    | 主要模型      | 学习目标           |
| ------- | --------- | -------------- |
| 人脸检测    | MTCNN     | 找到人脸位置         |
| 人脸识别    | FaceNet   | 判断两张人脸是否属于同一个人 |
| 人脸特征提取  | ArcFace   | 学习人脸特征向量       |
| 人脸关键点检测 | CNN、HRNet | 定位眼睛、鼻子、嘴巴等    |

重点学习：

- Embedding（特征向量）。
- Triplet Loss。
- Contrastive Loss。
- Cosine Similarity。
- Face Verification / Identification。

推荐项目：FaceNet 人脸识别。

它能帮助你理解 CNN 如何把图片转换成具有区分能力的特征向量。

## 5. 图像生成与图像风格转换

进阶

[DCGAN (Deep Convolutional GAN) – Naoki Shibuya](https://images.openai.com/static-rsc-4/RIOgE-gye-KPqNB5bmZyzboLlEEZeoGEIg9PipfLpDRSPMh2c4snHm2CuIcXFPVavMLrfmpPmJAqHd9MFtsnVfbY7qdCx4-UkaYsBW75EM1cTVoSL7ePktn3mK7w8xmLgyJVHRTAWNwrS70h-kBn7pqFLP_cPFIb5-LTEGAURv8?purpose=inline)

[pix2pix：使用条件 GAN 进行图像到图像的转换  |  TensorFlow Core](https://images.openai.com/static-rsc-4/7LmOBLCQP6EX7QkJLWg0JutF5gxCqVx0J_1jLlL1Z1VhFNo4E4Avg6rtyxbwFsy3AA9-LbH2mHQpQTCQOEGeDaOPNaDdlQJYcYJ2itYVxA01-zDk-cwavn9R_Jev2V4kTRrHfWdDgpTcQ-HdxQK3X44huYTrIRu_1i6ZCjGD2uE?purpose=inline)

[CycleGAN  |  TensorFlow Core](https://images.openai.com/static-rsc-4/AsDESYt44ytRDdMtvDSxz1YZnKmDuurvnwEDwNMinlm3DrKzN4ZLWi34YYhwE0EhNA9luYkwhlaYFYCDYEIGPa7C7b7UTNm8SR00VwRjQIog_qyKcUE-VGpf07B8dn-N7pdYw2Yy9_7gWPy4s8u5g6-6rZ-uCDXPH9UjPA6cmCg?purpose=inline)

| 经典项目    | 主要模型         | 学习目标          |
| ------- | ------------ | ------------- |
| 手写数字生成  | DCGAN        | 学习生成对抗网络      |
| 图像风格转换  | CycleGAN     | 将一种风格转换为另一种风格 |
| 图像到图像转换 | Pix2Pix      | 学习条件图像生成      |
| 超分辨率    | SRCNN、ESRGAN | 将低分辨率图片放大     |

重点学习：

- Generator（生成器）。
- Discriminator（判别器）。
- GAN Loss。
- Transposed Convolution。
- Perceptual Loss。

## 6. 其他经典工业项目

实践应用

[Surface Defect Detection of Hot Rolled Steel Based on Attention Mechanism and Dilated Convolution for Industrial Robots | MDPI](https://images.openai.com/static-rsc-4/Qjn1EAmK3gM6AQAiq5h-42fBt4trY9c3USTOb16A7gsmSpLPU3kjxpw1KZzo5kSqKpFtTqcChp5KrTEagrnkrn4PM4y264p9hPwoDIapTtq7Yr777WK9HYkPhdSEUVUZ32n0CrUSC3IEyoGr-86Re01aoMQOjk-St0Hoz90lBxs?purpose=inline)

工业缺陷检测

- 数据集：MVTec AD
- 模型：CNN、ResNet
- 识别产品表面缺陷

[Optical Character Recognition (OCR): Definition & How To Guide](https://images.openai.com/static-rsc-4/zZNKRQzmgSseasr_1xy8aU-F4DaXhZ8qzXG3a8KV3TLzyV1rNSWb7rW28XQK30cb3AiN4H6SFhVroaN0fyzB2KqTlTsky3tPM1u8-CGIUtj8NgDZtVnf-UVzJnQf3BttR3FDLKtA53HSHWDk_Eo-TUyPNFAHGtIRfXyjhQEMp6E?purpose=inline)

OCR 文字识别

- 数据集：MNIST、SVHN
- 模型：CNN、CRNN
- 识别图片中的文字

[Medium](https://images.openai.com/static-rsc-4/R-VErJBoZVFaJm8K-cMOWxxngvzFDo2APd9Z5RkzwNdp0Rdi5gTGFw2ZmqwgSAMLKMD0nNiGuBtApZ96rxJN9wsIXvOXiR6qnX3J8fWKXyjCxYrAH2OFei_RH-GDJycXeqKagtnupSUZVc1B7X1mdjWqmFwAbpMNLwigv4P7E80?purpose=inline)

医学影像分类

- 数据集：Chest X-Ray
- 模型：ResNet、DenseNet
- 图像辅助分类

[Counting Cars in Images | Computer Vision Projects | Codersarts AI](https://images.openai.com/static-rsc-4/1OQ0nxD-bGWu-7El2ZuRa6-P8kedO5CLwssg5iEzGA-7zf2an2UtY_uCtZ9YJsnajfXY_ErTnAOX8pRIRal9UNnu5maYFTCEycXf_rOeM65D9nGcD3-6M1JvK3PC-K4k8iwKSjbfpWVj8cbiRTcEsq6tVLAiZdW_MbDrqomk0c0?purpose=inline)

车辆检测与计数

- 数据集：UA-DETRAC
- 模型：YOLO
- 车辆定位与计数

注意：医学影像项目适合学习模型开发，不应将未经临床验证的模型预测用于实际诊断。

# 二、如果按照学习顺序，我建议你做这 10 个项目

上面列出了不同方向的经典项目，但不建议你一下子全部开始。结合你目前已经完成 CIFAR-10 CNN 分类的进度，我建议按照下面的路线继续。

| 顺序 | 项目                 | 推荐模型               | 难度    | 主要学习内容          |
| -- | ------------------ | ------------------ | ----- | --------------- |
| 1  | Fashion-MNIST 服装分类 | LeNet、简单 CNN       | ★     | CNN 基础巩固        |
| 2  | CIFAR-100 图像分类     | ResNet、VGG         | ★★    | 多分类、模型优化        |
| 3  | 猫狗图像分类             | VGG16、ResNet18     | ★★    | 迁移学习、微调         |
| 4  | 交通标志识别             | CNN                | ★★    | 实际图像分类          |
| 5  | 人脸识别               | FaceNet、ArcFace    | ★★★   | 特征向量、度量学习       |
| 6  | 工业缺陷检测             | ResNet、Autoencoder | ★★★   | 异常检测            |
| 7  | YOLO 目标检测          | YOLO               | ★★★★  | 目标定位、检测         |
| 8  | U-Net 医学图像分割       | U-Net              | ★★★★  | Encoder、Decoder |
| 9  | OCR 文字识别           | CRNN               | ★★★★  | CNN + RNN       |
| 10 | 图像超分辨率             | SRCNN、ESRGAN       | ★★★★★ | 图像重建、生成         |

## 三、每个阶段重点掌握什么？

我建议你把 CNN 学习分成下面几个阶段：

1. 阶段一：图像分类

   当前阶段

   项目：CIFAR-10、CIFAR-100、猫狗分类。

   掌握：Conv2d、MaxPool2d、BatchNorm、Dropout、ResNet、迁移学习。
2. 阶段二：目标检测

   项目：交通标志检测、行人检测、车辆检测。

   掌握：Bounding Box、IoU、NMS、YOLO、mAP。
3. 阶段三：图像分割

   项目：细胞分割、道路分割。

   掌握：U-Net、FCN、Dice Loss、IoU、Encoder-Decoder。
4. 阶段四：特征提取与图像生成

   项目：人脸识别、工业异常检测、图像超分辨率。

   掌握：Embedding、Triplet Loss、Autoencoder、GAN。

## 四、推荐使用的数据集

以下是这些项目常用的数据集，可以直接通过官方数据集页面或 PyTorch 数据集接口了解。

| 数据集           | 适合项目    | 参考资料                                                                                                             |
| ------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| MNIST         | 手写数字识别  | [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html?utm_source=chatgpt.com) |
| Fashion-MNIST | 服装分类    | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist?utm_source=chatgpt.com)                         |
| CIFAR-100     | 100 分类  | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html?utm_source=chatgpt.com)                                      |
| COCO          | 目标检测、分割 | [COCO](https://cocodataset.org/?utm_source=chatgpt.com)                                                          |
| Cityscapes    | 道路场景分割  | [Cityscapes](https://www.cityscapes-dataset.com/?utm_source=chatgpt.com)                                         |
| MVTec AD      | 工业异常检测  | [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad?utm_source=chatgpt.com)                      |

也可以查看 [Torchvision 官方数据集列表](https://docs.pytorch.org/vision/main/datasets.html?utm_source=chatgpt.com)，其中提供了分类、目标检测和图像分割等任务的数据集。

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

Torchvision main documentation+2



## 五、我的建议：你下一个项目做什么？



最推荐

&#x20;猫狗图像分类（Cats vs Dogs）+ ResNet18 迁移学习



[Image classification from scratch](https://images.openai.com/static-rsc-4/bXa7LmBXk_yWXuWW9APQDtzQZWQ0ZzfPhP7MAUsu75sRUMJTVkMfjcc5NccLCVNm6F7DPa_aRnMxXiyLSLSkwDkgiLs5JOn1571vvF0dZOZFLVp7LZHs6ckDUTQv6Soh8Ks_Rh7-p7FMKbl59nLQ5_iSHj2ZTFG-uZbXxa8nlRE?purpose=inline)

[Explainable Convolution Neural networks : Training ResNet-18 from scratch on ImageNet-200 dataset and explaining the predictions using Class Activation Maps | by Debayan Mitra | Medium](https://images.openai.com/static-rsc-4/2tz7cWMlatvBbLxiKqdt1mV488oZ2RV2bigaMeTOxfmi0X7V-qVyECKecbYE5hA4bl-w3XBrjYJ8pdkjT37y4oFlc_z0qSd3ne6eXIlgrEVfGWFWKhl8Ph2reGV_yT1Kp3GO0EdnBjn0Juc9WtLF7pMttKAd8wVK_6avMZfNjuk?purpose=inline)

[Reuse versus Adaptation: Decomposing the ImageNet-to-CIFAR-10 Transfer Win for ResNet-18](https://images.openai.com/static-rsc-4/GPmNU90IU0sZgEoJshlt3OQW_lVN8uTWhrRHTTNFHWLGDltX7RH1rG-gPqMMWXLezqa09EVLihyo81_3mDylCXbByhiIGSD2h-3DvERTApmztoLawhLAjcppjiOTMqMFLu6u2XCxB1Kjn6BDblo6VnytYxAOTVDyKyDwwC0Mi7Q?purpose=inline)

5

为什么推荐这个项目？

因为你已经使用 CIFAR-10 完成了一个从零构建 CNN 的训练流程，下一步应该学习实际项目中非常重要的技术：迁移学习（Transfer Learning）。

这个项目可以让你学习：

1. 如何加载 ImageNet 预训练模型。
2. 如何使用 ResNet18。
3. 如何冻结卷积层参数。
4. 如何替换最后的全连接层。
5. 如何只训练分类层。
6. 如何解冻部分网络进行 Fine-tuning。
7. 如何比较从零训练和迁移学习的效果。

这比继续做一个结构类似 CIFAR-10 的简单分类项目更有学习价值。

我的建议是：先完成猫狗分类，再进入 YOLO 目标检测。 这样能够从 CNN 的基础图像分类，逐步过渡到计算机视觉的实际应用。