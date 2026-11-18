# PyTorch TensorBoard：可视化训练过程从入门到实践

TensorBoard 是 TensorFlow 生态中发展起来的训练可视化工具，PyTorch 可通过 torch.utils.tensorboard.SummaryWriter 写入兼容的 event 日志文件。训练脚本负责记录数据，TensorBoard 读取日志并在浏览器中展示曲线、图像、模型图、参数分布等信息。

它不负责训练模型，也不会自动理解 loss。开发者需要明确记录什么指标、在哪个 step 记录、训练集与验证集如何区分。好的可视化能帮助回答：模型是否在学习、是否过拟合、梯度是否异常、不同实验哪个更好。

本文从原理和安装开始，逐步讲解标量、训练/验证曲线、图像、模型计算图、直方图、超参数实验、日志管理、检查点和排错。所有代码示例均配有概念说明，主线使用简单 MLP 分类任务。示例适用于现代 PyTorch 2.x；TensorBoard 相关包需另行安装。

## 1. 训练为什么需要可视化

训练时模型参数不断更新，只看最终准确率会丢失过程信息。例如训练 loss 一直不降，可能是学习率、数据、梯度或标签有问题；训练 loss 降而验证 loss 升，常表示过拟合；loss 突然变成 NaN，可能是数值不稳定。

```text
样本 -> 前向计算 -> loss -> backward -> optimizer 更新参数
                         |
                         +-- 按 step/epoch 写入事件日志
                                      |
                                      +-- TensorBoard 读取并绘图
```


```python
# 最简单的训练监控数据可以是一个 Python 数值
epoch = 1
train_loss = 0.82
valid_loss = 0.91
print(epoch, train_loss, valid_loss)
```

普通 print 只能查看文本输出且不便比较长时间实验。TensorBoard 将带标签的数值和其他内容保存到事件文件，再提供交互式图表。

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/demo")
writer.add_scalar("loss/train", 0.82, global_step=1)
writer.add_scalar("loss/valid", 0.91, global_step=1)
writer.close()
```

## 2. TensorBoard 的基本组成

核心组件有三部分：训练代码中的 SummaryWriter、磁盘上的 event 文件、浏览器里的 TensorBoard 服务。Writer 将 summary 数据异步写入日志目录；TensorBoard 服务读取目录并展示数据。日志目录不是模型 checkpoint，也不是训练代码本身。

```text
PyTorch 程序
  -> SummaryWriter.add_scalar/add_image/...
  -> runs/实验名/events.out.tfevents...
  -> tensorboard --logdir runs
  -> 浏览器面板
```

```python
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

log_dir = Path("runs/first_experiment")
writer = SummaryWriter(log_dir=log_dir)
writer.add_scalar("demo/value", 1.0, 0)
writer.flush()  # 立即把缓冲数据写出
writer.close()
print(log_dir.resolve())
```

Writer 通常会在目录中创建带时间和进程信息的 event 文件。不要手工编辑 event 文件；通过 Writer API 记录数据。

## 3. 安装与检查

PyTorch 的 SummaryWriter 通过 TensorBoard Python 包写 event 文件，因此需要安装 tensorboard。项目当前没有将它列为依赖，可在项目环境安装并选择是否写入 pyproject 配置。

```bash
uv add tensorboard
```

普通 pip 环境可使用：

```bash
python -m pip install tensorboard
```

检查两个包的版本与导入路径，确保命令使用的 Python 和运行训练脚本的 Python 是同一个环境。

```python
import torch
import tensorboard
from torch.utils.tensorboard import SummaryWriter

print("torch:", torch.__version__)
print("tensorboard:", tensorboard.__version__)
print(SummaryWriter)
```

若导入时报错，优先检查 tensorboard 是否装入当前虚拟环境。TensorBoard 命令行工具也应由同一环境启动，例如使用 uv run。

```bash
uv run tensorboard --version
```

## 4. 第一个 SummaryWriter

SummaryWriter 接收日志目录，可选 flush_secs、max_queue、filename_suffix 等参数。日志通常放在 runs/项目名/实验名；不同实验用不同子目录，避免混到同一条曲线。

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/mlp/baseline")
writer.add_scalar("accuracy", 0.75, global_step=0)
writer.close()
```

step 通常是全局训练步数，也可以按 epoch 记录，但同一指标应始终采用同一种横轴单位。混用 epoch 和 batch step 会使图表难以解释。

```python
writer = SummaryWriter("runs/demo_epoch")
for epoch in range(3):
    metric = 1.0 / (epoch + 1)
    writer.add_scalar("loss/train", metric, global_step=epoch)
writer.close()
```

训练中可显式 flush，确保刚写入的数据尽快可见；频繁 flush 会带来 I/O 开销，通常不必每个 batch 都调用。

```python
writer = SummaryWriter("runs/flush_demo", flush_secs=30)
writer.add_scalar("loss/train", 0.5, 1)
writer.flush()
writer.close()
```

## 5. 启动 TensorBoard 查看日志

训练完成或运行中，从项目目录启动 TensorBoard，并让 logdir 指向 event 文件所在目录的父级。Web 服务默认端口通常为 6006。

```bash
uv run tensorboard --logdir runs --port 6006
```

也可以使用当前 Python 环境的模块入口：

```bash
python -m tensorboard.main --logdir runs
```

打开终端输出的网址（通常是 http://localhost:6006）。如果端口已占用，可指定其他端口；远程服务器需按安全策略配置端口转发，不要在不可信网络上暴露服务。

```bash
uv run tensorboard --logdir runs --port 6007
```

## 6. 记录标量 add_scalar

标量是最常见的训练信息，如 loss、accuracy、learning rate。add_scalar(tag, scalar_value, global_step) 的 tag 是图表名称/分组路径，value 是单个数值，step 是横轴位置。

```python
writer = SummaryWriter("runs/scalar_demo")
for step in range(5):
    writer.add_scalar("loss/train", 1.0 / (step + 1), step)
    writer.add_scalar("accuracy/train", step / 5, step)
writer.close()
```

tag 中斜杠会形成分组，例如 loss/train 和 loss/valid 可放在 Loss 分组中。使用稳定、简明、统一的命名，不要每一步都生成不同 tag。

```python
metrics = {"loss/train": 0.35, "accuracy/train": 0.91}
for name, value in metrics.items():
    writer.add_scalar(name, value, global_step=100)
```

传入 Tensor 时先 detach 并转为 Python 数值，避免日志操作无意间保留计算图或传入多元素 Tensor。

```python
loss = torch.tensor(0.42, requires_grad=True)
writer.add_scalar("loss/example", loss.detach().item(), global_step=1)
```

## 7. Global step 的含义

global_step 应随训练进度单调增加。若按 batch 写入，常用 epoch * steps_per_epoch + batch_index；若按 epoch 写入，直接使用 epoch 编号。step 重复或回退可能导致图表点覆盖、重排或显示异常。

```python
epochs = 3
steps_per_epoch = 4
for epoch in range(epochs):
    for batch_idx in range(steps_per_epoch):
        global_step = epoch * steps_per_epoch + batch_idx
        writer.add_scalar("debug/step", global_step, global_step)
```

训练和验证曲线可分别按 epoch 记录，这样横轴更直观。若每个 epoch 长短不同，按 batch 记录则能看到更细粒度变化。

```python
for epoch in range(3):
    train_loss = 1.0 / (epoch + 1)
    valid_loss = 1.2 / (epoch + 1)
    writer.add_scalar("loss/train_epoch", train_loss, epoch + 1)
    writer.add_scalar("loss/valid_epoch", valid_loss, epoch + 1)
```

## 8. 需要记录哪些指标

分类任务通常记录 train/validation loss、accuracy、learning rate；不平衡分类还应记录 precision、recall、F1 或 PR-AUC。回归任务可记录 MAE、RMSE；生成任务可记录 token loss、吞吐量等。记录指标应围绕诊断问题，不要因能记录而无限堆指标。

```python
metrics = {
    "loss/train": 0.42,
    "loss/valid": 0.51,
    "accuracy/train": 0.88,
    "accuracy/valid": 0.84,
    "learning_rate": 1e-3,
}
for tag, value in metrics.items():
    writer.add_scalar(tag, value, global_step=1)
```

训练集与验证集必须使用清晰不同的 tag。否则图上无法判断泛化差距，也可能误把训练表现当成验证表现。

```python
writer.add_scalars(
    "loss",
    {"train": 0.4, "valid": 0.6},
    global_step=1,
)
```

add_scalars 会把一组数值画在同一图上；也可用独立的 loss/train 和 loss/valid tag，便于 TensorBoard 按目录分组。

```python
writer.add_scalar("loss/train", 0.4, 1)
writer.add_scalar("loss/valid", 0.6, 1)
```

## 9. 训练时记录 batch 与 epoch 指标

batch loss 变化较大，适合观察优化是否稳定；epoch 平均 loss 更适合比较整体趋势。不能把最后一个 batch loss 当作整个 epoch 的训练 loss，batch 样本数不同时应按样本数加权累计。

```python
loss_sum = 0.0
sample_count = 0

for features, labels in train_loader:
    logits = model(features)
    loss = criterion(logits, labels)
    count = labels.size(0)
    loss_sum += loss.item() * count
    sample_count += count

epoch_loss = loss_sum / sample_count
writer.add_scalar("loss/train_epoch", epoch_loss, epoch)
```

如要记录 batch loss，global_step 应跨 epoch 连续增长。训练循环开始时先确定一个全局 step 变量，每消费一个 batch 后递增。

```python
global_step = 0
for epoch in range(num_epochs):
    for features, labels in train_loader:
        loss = train_one_batch(features, labels)
        writer.add_scalar("loss/train_batch", loss, global_step)
        global_step += 1
```

## 10. 一个可运行的 MLP 分类例子

下面用合成二维数据演示完整训练、验证和 TensorBoard 记录，避免下载数据集。模型预测 3 个类别；CrossEntropyLoss 接收 logits [B,C] 和 long 类别标签 [B]。

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_data(n):
    centers = torch.tensor([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
    labels = torch.randint(0, 3, (n,))
    features = centers[labels] + 0.7 * torch.randn(n, 2)
    return features, labels

train_x, train_y = make_data(900)
valid_x, valid_y = make_data(300)
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=128)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.layers(x)

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
```

训练期间按 batch 记录 loss 和按 epoch 记录平均 loss/accuracy。验证集使用 eval 和 inference_mode，不能反向传播更新参数。

```python
from pathlib import Path

writer = SummaryWriter("runs/mlp/synthetic_classification")
global_step = 0
num_epochs = 10

try:
    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            count = labels.size(0)
            train_loss_sum += loss.item() * count
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_count += count
            writer.add_scalar("loss/train_batch", loss.item(), global_step)
            global_step += 1

        train_loss = train_loss_sum / train_count
        train_accuracy = train_correct / train_count

        model.eval()
        valid_loss_sum = 0.0
        valid_correct = 0
        valid_count = 0
        with torch.inference_mode():
            for features, labels in valid_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                count = labels.size(0)
                valid_loss_sum += loss.item() * count
                valid_correct += (logits.argmax(1) == labels).sum().item()
                valid_count += count

        valid_loss = valid_loss_sum / valid_count
        valid_accuracy = valid_correct / valid_count
        writer.add_scalar("loss/train_epoch", train_loss, epoch + 1)
        writer.add_scalar("loss/valid_epoch", valid_loss, epoch + 1)
        writer.add_scalar("accuracy/train", train_accuracy, epoch + 1)
        writer.add_scalar("accuracy/valid", valid_accuracy, epoch + 1)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
        print(epoch + 1, train_loss, valid_loss, valid_accuracy)
finally:
    writer.close()
```

示例中 train/validation 数据由同一分布独立生成，仅用于演示 API；真实项目应使用合理的数据切分，且验证集不得参与梯度更新。

```python
# 训练结束后模型参数和 TensorBoard 日志是两类不同产物
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": num_epochs,
}
torch.save(checkpoint, "mlp_checkpoint.pt")
```

## 11. 用 TensorBoard 观察过拟合

训练 loss 下降而验证 loss 开始上升，常是过拟合信号。TensorBoard 让两条曲线在同一横轴下可视比较；但这是诊断线索，不是单独的因果证明，还要结合数据切分和指标定义。

```python
writer.add_scalar("loss/train", 0.3, epoch)
writer.add_scalar("loss/valid", 0.7, epoch)
```

若验证 loss 最低点出现在早期，应保存该 epoch 的最佳 checkpoint，而不是默认保存最后一次参数。TensorBoard 曲线可辅助确定最佳 epoch，验证逻辑应在训练代码中自动实现。

```python
best_valid_loss = float("inf")
if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), "best_model.pt")
```

## 12. 记录学习率

学习率调度器会改变每一阶段的更新幅度。记录 optimizer 当前 param_groups 中的学习率，可以解释 loss 曲线拐点或训练停滞。

```python
for group_index, group in enumerate(optimizer.param_groups):
    writer.add_scalar(
        f"learning_rate/group_{group_index}",
        group["lr"],
        global_step=epoch + 1,
    )
```

使用 scheduler 时，按训练设计在正确时机调用 step，并在调用前后明确记录哪一个学习率。ReduceLROnPlateau 通常依据验证指标调用。

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
for epoch in range(10):
    train_one_epoch()
    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
    scheduler.step()
```

## 13. 记录图片 add_image

add_image 用于写入图像，例如样本输入、重建结果或生成样本。默认数据格式通常是 CHW；若数据为 HWC，需要指定 dataformats。记录前要将 Tensor 移到 CPU，并确保范围和 dtype 符合可视化预期。

```python
image = torch.rand(3, 28, 28)  # [C,H,W]，值在 [0,1]
writer.add_image("sample/image", image, global_step=0)
```

批量图像可用 add_images，输入常为 NCHW。TensorBoard 也可用 add_image 写入 torchvision.make_grid 生成的拼图，便于一次查看一批样本。

```python
from torchvision.utils import make_grid

images = torch.rand(16, 1, 28, 28)
grid = make_grid(images, nrow=4, normalize=True)
writer.add_image("samples/grid", grid, global_step=0)
```

对真实数据可同时记录输入与模型输出，核对数据管线、归一化和重建质量。不要每个 step 都写大量图片，以免日志膨胀。

```python
input_image = torch.rand(1, 28, 28)
reconstructed = torch.rand(1, 28, 28)
writer.add_image("reconstruction/input", input_image, 1)
writer.add_image("reconstruction/output", reconstructed, 1)
```

## 14. 记录模型计算图 add_graph

add_graph 会用示例输入运行模型追踪计算图，帮助查看模型模块连接和张量流向。它不是训练动态图的完整调试器；模型需要能接受示例输入，控制流和某些动态行为可能影响追踪。

```python
example_input = torch.randn(1, 2).to(device)
writer.add_graph(model, example_input)
```

模型应处于适合追踪的状态，并在写图之后继续正常训练。若 forward 有随机增强、依赖 Python 数据的分支或多个输入，要准备符合真实接口的示例参数。

```python
class TwoInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 2)

    def forward(self, features, extra):
        return self.linear(features + extra)

model_two = TwoInputModel()
example_features = torch.randn(1, 5)
example_extra = torch.randn(1, 5)
writer.add_graph(model_two, (example_features, example_extra))
```

## 15. 参数直方图 add_histogram

add_histogram 可以记录模型参数或梯度的分布随训练的变化。它适合诊断参数是否爆炸、长期集中在零附近或梯度是否异常，但会生成较多数据，不必每个 batch 都记录。

```python
for name, parameter in model.named_parameters():
    writer.add_histogram(f"parameters/{name}", parameter.detach().cpu(), epoch)
    if parameter.grad is not None:
        writer.add_histogram(f"gradients/{name}", parameter.grad.detach().cpu(), epoch)
```

如果在 optimizer.step 后记录梯度，注意梯度仍可能保留，也可能随后清零；更清楚的时机通常是在 backward 后、step 前。参数直方图则可以在 step 后观察更新结果。

```python
optimizer.zero_grad(set_to_none=True)
logits = model(features)
loss = criterion(logits, labels)
loss.backward()

for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        writer.add_histogram(f"gradients/{name}", parameter.grad.detach(), global_step)

optimizer.step()
```

## 16. 记录文本 add_text

add_text 可写入实验说明、配置摘要或少量样例。它支持 Markdown 展示，适合记录可读信息；不要写入隐私文本、密钥或敏感数据。

```python
writer.add_text("experiment/notes", "MLP baseline with AdamW", global_step=0)
writer.add_text("experiment/config", "| key | value |\\n|---|---|\\n| lr | 0.01 |", 0)
```

可记录模型配置、数据版本和随机种子，使日志目录更具可追溯性。

```python
config = {"seed": 42, "batch_size": 64, "lr": 0.01, "hidden": 32}
writer.add_text("experiment/config", str(config), global_step=0)
```

## 17. 超参数实验 add_hparams

add_hparams 可将超参数与最终指标关联，TensorBoard 的 HParams 面板可帮助比较不同实验。每次实验应使用独立 run 目录，配置名和值类型保持一致。

```python
writer = SummaryWriter("runs/hparams/lr_001")
writer.add_hparams(
    {"lr": 0.01, "batch_size": 64, "optimizer": "AdamW"},
    {"hparam/valid_accuracy": 0.91, "hparam/valid_loss": 0.28},
)
writer.close()
```

真实调参应自动循环运行多个配置并记录结果。不要只在 TensorBoard 里挑一个高分就忽略验证集过拟合或测试集泄漏。

```python
for lr in [1e-2, 1e-3]:
    run_name = f"runs/lr_{lr:g}"
    writer = SummaryWriter(run_name)
    writer.add_hparams({"lr": lr}, {"final/valid_accuracy": 0.8})
    writer.close()
```

## 18. 直方图与稀疏记录

图像、直方图、embedding 等数据可能很大，应选择记录频率。比如每个 epoch 记录参数直方图、每若干 epoch 记录图片，标量则可按 batch 或 epoch 记录。

```python
if (epoch + 1) % 5 == 0:
    for name, parameter in model.named_parameters():
        writer.add_histogram(f"parameters/{name}", parameter.detach(), epoch + 1)
```

SummaryWriter 的 max_queue 和 flush_secs 控制缓冲与写盘频率；队列更大可降低同步 I/O，但异常中断时未落盘数据可能丢失。正常退出一定 close。

```python
writer = SummaryWriter(
    log_dir="runs/buffer_demo",
    max_queue=20,
    flush_secs=15,
)
```

## 19. 组织日志目录

日志目录结构决定 TensorBoard 如何比较实验。可按项目/数据/模型/配置命名，每个训练 run 使用唯一目录；同一实验反复写入旧目录会造成曲线混合或 step 冲突。

```python
from datetime import datetime
from pathlib import Path

run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = Path("runs") / "mlp" / run_name
writer = SummaryWriter(log_dir=str(log_dir))
print(log_dir)
```

命令行指定 runs 父级时，TensorBoard 会读取其子目录中的 runs。目录名通常作为实验标签，可表达关键差异，但避免过长和包含隐私信息。

```bash
uv run tensorboard --logdir runs/mlp
```

## 20. 关闭 writer 和异常安全

Writer 会缓冲数据；close 会刷新并释放资源。训练中断或异常时，建议用 try/finally 或上下文管理确保关闭。仅靠进程退出可能丢失最后一段尚未 flush 的事件。

```python
writer = SummaryWriter("runs/safe_close")
try:
    writer.add_scalar("loss/train", 1.0, 0)
    # 执行训练
finally:
    writer.close()
```

短脚本可用上下文管理器表达资源生命周期：

```python
with SummaryWriter("runs/context_manager") as writer:
    writer.add_scalar("demo/value", 1.0, 0)
```

## 21. 保存模型与日志不是一回事

TensorBoard event 文件记录可视化数据，不包含可用于推理的完整模型参数；checkpoint 保存模型状态（可选还包含 optimizer、epoch 等）。正式实验应两者分别保存，并确保目录结构和配置可追溯。

```python
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "checkpoint.pt",
)
```

加载模型时先恢复结构，再加载 state_dict；TensorBoard 不会自动帮你恢复训练。

```python
checkpoint = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

## 22. 从日志推断训练状态

训练 loss 和验证 loss 都下降，通常说明当前阶段仍在学习；训练 loss 继续下降而验证 loss 上升，提示过拟合；两条都不动，优先查学习率、梯度、数据和标签；曲线剧烈振荡，可能是学习率太大、batch 太小或数据异常。

```python
# 示例：记录两条验证趋势用于观察，不代表阈值规则
for epoch, train_loss, valid_loss in [
    (1, 1.2, 1.3), (2, 0.8, 0.9), (3, 0.5, 1.0)
]:
    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/valid", valid_loss, epoch)
```

TensorBoard 展示的是程序记录的数值，不会判断累计方式是否正确。loss 应按样本数正确聚合，accuracy 的分母要与计数范围一致，验证指标应在 eval/inference_mode 下计算。

```python
# 不同大小 batch 的 loss 按样本加权平均
total_loss = sum(batch_loss * batch_size for batch_loss, batch_size in [(0.2, 64), (0.4, 16)])
total_count = 64 + 16
epoch_loss = total_loss / total_count
print(epoch_loss)
```

## 23. 常见故障排查

常见问题包括 TensorBoard 命令找不到、页面没有数据、图表出现重复曲线、step 乱序、图像显示范围不对，以及 add_graph 输入不匹配。

```python
from pathlib import Path

log_dir = Path("runs")
event_files = list(log_dir.rglob("events.out.tfevents.*"))
print("event files:", len(event_files))
for path in event_files[:5]:
    print(path)
```

| 问题 | 常见原因 | 检查方式 |
| --- | --- | --- |
| SummaryWriter import 失败 | 没安装 tensorboard 或环境不一致 | 在训练环境安装并检查解释器 |
| TensorBoard 页面空白 | logdir 指错、writer 未 flush/close、无事件文件 | 搜索 event 文件并检查启动路径 |
| 指标曲线断裂/乱序 | global_step 重复或回退 | 使用单调递增 step，分 run 写日志 |
| 多次训练曲线混在一起 | 重用旧目录 | 每次 run 创建独立目录 |
| 图像黑/白或颜色异常 | 范围、dtype、CHW/HWC 不匹配 | 检查值域和 dataformats |
| add_graph 报输入错误 | 示例输入 shape/device 不匹配 | 用与 forward 相同的输入结构测试 |
| 日志增长很快 | 每步写图像/直方图或 embedding | 降低频率、清理非必要 run |

查看本次运行的 event 路径和训练进程使用的 logdir 是否一致：

```bash
find runs -name 'events.out.tfevents.*'
```

## 24. 使用 TensorBoard 做实验比较

比较实验时，要保证数据切分、评估实现和指标定义一致，仅改变要研究的变量。例如学习率实验每个配置用独立目录，最后用验证指标比较；测试集只在模型方案确定后评估。

```python
learning_rates = [1e-2, 1e-3, 1e-4]
for lr in learning_rates:
    run_dir = f"runs/lr_sweep/lr_{lr:g}"
    writer = SummaryWriter(run_dir)
    writer.add_text("config", f"learning_rate={lr}", 0)
    # 真正实验中在此训练模型，并逐 epoch 写验证指标
    writer.close()
```

同一指标使用相同 tag，可以在 TensorBoard 的 runs 面板中叠加比较；对实验名称和配置做稳定记录。

```python
for run_name, final_accuracy in [("baseline", 0.82), ("dropout", 0.86)]:
    writer = SummaryWriter(f"runs/compare/{run_name}")
    writer.add_scalar("metrics/valid_accuracy", final_accuracy, 10)
    writer.close()
```

## 25. 与其他可视化工具的分工

TensorBoard 长于训练曲线、事件日志、图像和实验对比；Matplotlib 适合代码中自定义静态图；Weights & Biases 等实验平台提供远程协作、表格和云端追踪。选择取决于项目环境与数据政策，不能把日志上传到未经批准的服务。

```python
# TensorBoard 记录数值的核心接口保持简单
writer.add_scalar("loss/train", 0.25, global_step=100)
writer.add_scalar("metrics/valid_f1", 0.88, global_step=100)
```

## 26. 推荐实践清单

记录前先回答：指标定义是什么、横轴单位是什么、训练/验证如何区分、多久记录一次、敏感信息是否会写入日志。日志和 checkpoint 分开保存，run 配置能够追溯。

```python
run_config = {
    "model": "MLP",
    "optimizer": "AdamW",
    "learning_rate": 1e-2,
    "batch_size": 64,
    "seed": 42,
    "step_unit": "epoch",
}
writer.add_text("config/json", str(run_config), global_step=0)
```

训练循环中建议至少记录以下标量：

```python
writer.add_scalar("loss/train", train_loss, epoch)
writer.add_scalar("loss/valid", valid_loss, epoch)
writer.add_scalar("accuracy/train", train_accuracy, epoch)
writer.add_scalar("accuracy/valid", valid_accuracy, epoch)
writer.add_scalar("learning_rate", current_lr, epoch)
```

## 27. 总结

TensorBoard 可视化训练的基本过程是：安装依赖，建立唯一日志目录，创建 SummaryWriter，在明确的 step 上记录带 tag 的标量/图像/模型信息，关闭 writer，再启动 TensorBoard 指向日志父目录。

```text
训练数据 -> 模型与 loss -> 训练代码显式记录 summary -> event 文件
                                                    -> TensorBoard 面板
```

TensorBoard 只负责存储和呈现记录，不会修正指标统计错误、不负责保存模型、不判断实验是否公平。让日志可信的关键仍是正确的训练/验证流程、稳定的 step、准确聚合的指标和可复现配置。
