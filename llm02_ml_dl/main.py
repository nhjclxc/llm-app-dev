def main():
    print("Hello from llm02-ml-dl!")




if __name__ == "__main__":
    main()


import torch

print("PyTorch 版本：", torch.__version__)
print("CUDA 是否可用：", torch.cuda.is_available())
print("MPS 是否可用：", torch.backends.mps.is_available())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("运行设备：", device)
# 创建张量并进行计算
x = torch.tensor([1, 2, 3], device=device)
print(x)
print(x * 2)



x = torch.randn(1000, 1000, device=device)
y = x @ x

print(x.shape)
print(y.shape)
