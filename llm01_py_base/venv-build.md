
```shell

## 1️⃣ 初始化项目
uv init model_name
cd model_name
## 2️⃣ 创建虚拟环境
uv venv --python 3.11
## 3️⃣ 激活环境
# Windows 
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
## 4️⃣ 安装依赖
### 快速构建 机器学习模型、API 或任何任意 Python 函数的演示或 Web 应用程序。 
uv add gradio
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
