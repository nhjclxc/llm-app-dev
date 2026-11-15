# BERT

BERT全称叫做Bidirectional Encoder Representations from Transformers, 论文地址: [[1810.04805] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (arxiv.org)](https://arxiv.org/abs/1810.04805)

BERT是谷歌AI研究院在2018年10月提出的一种预训练模型. BERT本质上就是Transformer模型的encoder部分, 并且对encoder做了一些改进.

* 官方代码和预训练模型
  Github: [https://github.com/google-research/bert](https://github.com/google-research/bert)

下图中编码器部分即BERT的基本结构.

![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/533/1666272184013/2ab4c5bfc77b4e86a7e9015ebf606445.png)

BERT对Transformer的encoder部分做了哪些改进呢:

1. 模型更大, 训练参数更多.
   1. 两个版本: Base: blocks=12, hidden_size=768, heads=12, parameters=110M
   2. Large: blocks=24, hidden_size=1024, heads=16, parameter=340M
   3. 在大规模数据上训练, 30亿个词的数据集
2. 位置编码变成可以学习得到.
3. 每个输入样本是一个句子对.
4. 加入了额外的片段嵌入(如下图):

![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/533/1666272184013/cc56e5b271cd4344a97309c40f22d359.png)

因为BERT是一个预训练模型, 需要提前训练好. BERT提出了两个预训练任务:

1. 带掩码的语言模型(Masked Language Model): 编码双向上下文
   1. 随机(15%概率)将一些词元换成&#x3c;mask>
   2. 因为微调任务中没有&#x3c;mask>
      1. 80%概率将选中的词换成&#x3c;mask>
      2. 10%概率换成随机词元
      3. 10%概率保持原有词元
2. 下一句子预测(Next Sentence Prediction):预测一个句子对中两个句子是否相邻, 建模文本对之间的逻辑关系
   1. 训练样本中50%概率选择相邻句子对
   2. 50%概率选择随机句子对.

**学习建议**

bert难理解，主要在于前期知识积累不够。如果懂transformer，那么理解bert的模型框架就不是问题，顶多需要理解的是预训练的任务及过程。那么，学习bert的正确路线是什么？以下：

seq2seq机器翻译模型 > attention机制 > transformer > bert模型及预训练方式 > bert下游任务微调
