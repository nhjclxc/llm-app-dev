# GPT

### 1. GPT的前世今生

![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/533/1676291113019/8d8fd0d12eff4a35b5fd2b0bcae0eaaa.png)

- Transformer是一个用作翻译任务的模型，谷歌出品。
- GPT全称 lmproving Language Understanding by Generative Pre-Training，用预训练语言理解模型。OPENAI出品。
- BERT全称Pre-training of Deep BidirectionalTransformers for Language Understanding，双向预训练语言理解模型。谷歌出品。GPT与BERT都是基于Transformer的模型结构。
- BERT在学术界比GPT要火很多，BERT论文引用量是58.9K。GPTI.,2.3加起来的论文引用量是17.7K。(截至2023年2月10日)
- GPTI，2，3本质没有区别,仅是模型参数的数量越来越多。
- GPT3.5是GPT3在不同任务上微调之后的综合模型。
- ChatGPT则是利用GPT系列模型封装的智能聊天Al应用。

### 2. GPT和BERT的区别

1. 模型结构:

- GPT是一种单向的 Transformer模型，只考虑一个词的左侧上下文。
- BERT是一种双向的Transformer模型，同时处理一个词的左侧和右侧的上下文。

![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/533/1676291113019/d4ffb6007a9347cbaf91207d8e60a7dc.png)

2. 预训练任务:

- GPT的预训练任务是Language Modeling，它更注重语言生成。(简单理解，就是文本生成，如写作，写歌词等)
- BERT的预训练任务是Masked Language Modeling 和Next Sentence Prediction，它更注重语言理解和关系
  推断。(简单理解，就是完型填空，与预测B句子是否在A句子后面这两个任务联合训练。)

3. 应用领域:

- GPT则更专注于语言生成，如文本生成、对话生成等。
- BERT的应用领域更广泛，可以用于命名实体识别、问答系统、句子相似度等任务。

4. 模型参数量对比

![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/533/1676291113019/ae9e42036a374ddd93b4b48959ddf57a.png)

- BERT-base就是故意做成GPT的大小从而与之比较,结果效果上完胜。
- GPT2代开始将模型做大，BERT便不再与GPT竞争。

5. GPT半开源，BERT全开源。这意味着，做产品时可以把BERT模型部署在自己的服务器上从而脱离其母公司，而GPT不可以。
6. GPT目前有API可调用，BERT没有。这意味着对于普通程序员而言，基于GPT做产品的门槛远低于BERT
7. GPT的微调不是真正的微调,而是few-shot。

### 3. 微调和Few-Shot

1. 微调
   优势:训练效果与普通模型训练没有区别。预训练模型的存在相当于为此次训练做了大量的准备工作。
   劣势:所需算力与数据量与普通模型训练没有区别。

   ![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/533/1676291113019/5850362415294846b26c473f26fafcf2.png)
2. Few-Shot

优势:训练省力，计算量小,不需要太多数据。

劣势:容易过拟合。很考验预训练模型的泛化能力。

![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/533/1676291113019/753a430e720d4df3a6560a2ccad805d0.png)
