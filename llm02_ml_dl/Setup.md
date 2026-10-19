
# 0. init env

```shell
## 1️⃣ 初始化项目
uv init llm02_ml_dl
cd llm02_ml_dl
## 2️⃣ 创建虚拟环境
uv venv --python 3.11
## 3️⃣ 激活环境
# Windows 
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
## 4️⃣ 安装依赖
uv add pytest
uv add numpy pandas matplotlib
uv add jupyterlab jupyterlab-lsp python-lsp-server[all] jupyter-ai jupyterlab-variableinspector jupyterlab-code-formatter
# 安装 pytorch
# scikit-learn 机器学习包，joblib 保存训练好的机器学习模型
uv add scikit-learn joblib kaggle
uv add torch torchvision
# 安装深度学习加速模块 cdDNN（仅gpu环境需要安装）
uv add nvidia-cuda-runtime-cu12
uv add nvidia-cudnn-cu12
## 5️⃣ 运行代码
uv run python main.py
## 6️⃣ 输出当前依赖
uv pip freeze > requirements.txt
## 锁定依赖
uv lock
## 安装依赖
pip install -r requirements.txt
uv pip install -r requirements.txt

```

项目下所有ml、dl相关模型代码使用 pytorch 框架实现


启动 jupyterlab
```shell
uv run jupyter lab
```


验证 pytorch 是否可用
```python
import torch

print("PyTorch 版本：", torch.__version__)
print("CUDA 是否可用：", torch.cuda.is_available())
print("MPS 是否可用：", torch.backends.mps.is_available())
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("运行设备：", device)

# 创建张量并进行计算
x = torch.tensor([1, 2, 3], device=device)
print(x * 2)

```

配置 kaggle cli token，进入[kaggle官网](https://www.kaggle.com/settings/api)生成 api token


# 1. 寻找数据集网站

## 1.1搜索网站
- [kaggle](https://www.kaggle.com/datasets)
- [阿里天池](https://tianchi.aliyun.com/dataset)
- [google datase tsearch](https://datasetsearch.research.google.com/)
- [huggingface](https://huggingface.co/papers/trending)
- [pytorch框架内置书记集](https://docs.pytorch.org/vision/stable/datasets.html)
- [openml](https://www.openml.org/)
- [uci edu](https://archive.ics.uci.edu/datasets/)
- [DataCastle](https://www.datacastle.cn/dataset_list.html)
- [AwesomeData](https://awesomedataworld.slack.com/)
- [aws](https://registry.opendata.aws/)
- [awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets)
- [data.world](https://data.world/)
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- [visualdata](https://visualdata.io/)
- [nlpir.org](http://www.nlpir.org/?action-category-catid-28)
- []()


## 1.2 其他
 - [文章1](https://tianchi.aliyun.com/forum/post/77400)
 - [文章2, 包含ml，dl，nlp等](https://www.zhihu.com/tardis/bd/ans/2477782053)
 - [50个最佳机器学习公共数据集](https://github.com/XNoteW/Studying/blob/master/dataset.md)
 - [文章3](https://blog.csdn.net/weixin_42010722/article/details/120199613)
 - []()
 - []()


## 1.3 具体数据集
### 波士顿房价

[波士顿房价](https://www.kaggle.com/datasets/altavish/boston-housing-dataset)，

[示例代码](https://www.kaggle.com/code/chenjiaxin0/notebook8ea22606fe)

[波士顿房价预测](https://www.kaggle.com/competitions/dl-bhpp)


```shell
curl -L -o ./boston-housing-dataset.zip https://www.kaggle.com/api/v1/datasets/download/altavish/boston-housing-dataset
```

### 波士顿房价
heart-disease.csv
[波士顿房价](https://www.kaggle.com/datasets/altavish/boston-housing-dataset)，

[示例代码](https://www.kaggle.com/code/chenjiaxin0/notebook8ea22606fe)

[波士顿房价预测](https://www.kaggle.com/competitions/dl-bhpp)


```shell
curl -L -o ./boston-housing-dataset.zip https://www.kaggle.com/api/v1/datasets/download/altavish/boston-housing-dataset
```

### 手写数字体识别

[手写数字体识别](https://www.kaggle.com/datasets/hojjatk/mnist-dataset): [文件链接]() 建议直接使用使用：from torchvision.datasets import MNIST

```shell
curl -L -o ./mnist-dataset-dataset.zip https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
```





