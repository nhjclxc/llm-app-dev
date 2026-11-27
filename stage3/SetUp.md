


```shell
## 1️⃣ 初始化项目
uv init stage3
cd stage3
## 2️⃣ 创建虚拟环境
uv venv --python 3.11
## 3️⃣ 激活环境
# Windows 
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
## 4️⃣ 安装依赖
uv add numpy
# 测试包
uv add pytest
# 用于读取.env文件里面的key
uv add python-dotenv
## openai 官方客户端，tiktoken 官方编码器
uv add openai tiktoken
# 向量数据库 chromadb
uv add chromadb
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
