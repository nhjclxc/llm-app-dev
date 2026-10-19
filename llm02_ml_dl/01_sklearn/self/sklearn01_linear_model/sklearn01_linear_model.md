# sklearn 线性模型学习路线：从 0 到 1

> 目标：能够判断一个问题是否适合线性模型，独立完成数据预处理、训练、评估、解释、调参和部署前检查。
>
> 说明：本文中的“线性”指**对参数线性**。例如多项式回归虽然对输入变量呈曲线，但对参数仍是线性的，因此仍可用线性模型求解。

## 1. 学完后应该达到什么程度

完成本路线后，至少应能回答下面的问题：

1. 当前任务是回归还是分类？目标变量的类型、形状和业务代价是什么？
2. 为什么要划分训练集、验证集和测试集？为什么标准化只能在训练集上 `fit`？
3. 什么时候使用 `LinearRegression`，什么时候使用 `Ridge`、`Lasso` 或 `ElasticNet`？
4. 线性回归的系数、截距和 `R^2` 是否具有可信的业务解释？多重共线性会带来什么问题？
5. 分类任务中，为什么 `LogisticRegression` 是线性分类器？如何处理类别不平衡和阈值选择？
6. 训练误差很小但测试误差很大时，应该从数据泄漏、特征、正则化还是评估方式开始排查？
7. 如何用 `Pipeline` 和 `ColumnTransformer` 让预处理与模型一起交叉验证，避免线上训练/推理不一致？

## 2. 先修知识与工具

### 2.1 Python 和科学计算

- Python：函数、类、异常、文件读写、虚拟环境。
- NumPy：数组 `shape`、索引、广播、矩阵乘法、均值和方差。
- pandas：读取数据、缺失值、类型转换、筛选、分组、连接。
- Matplotlib/Seaborn：散点图、残差图、直方图、混淆矩阵。
- Jupyter Notebook 与命令行运行脚本。

### 2.2 数学最低要求

- 一元函数、偏导数和梯度的直观含义。
- 向量、矩阵、转置、秩、内积和范数。
- 均值、方差、协方差、相关系数。
- 概率基础：条件概率、期望、方差、正态分布和对数。
- 最小二乘思想：让预测值与真实值的残差平方和尽可能小。

不需要先把所有数学推导学完。建议每学一个模型，都做到“先用 API 跑通，再用公式解释，最后用一个小数据集验证”。

## 3. 统一的 sklearn 工作流

线性模型的学习始终围绕下面这条链路展开：

```text
明确目标和数据口径
    -> 划分 train/validation/test
    -> 识别数值列、类别列和缺失值
    -> 预处理（只在训练数据 fit）
    -> 训练基线模型
    -> 用合适指标评估
    -> 诊断偏差、方差、残差和泄漏
    -> 交叉验证与调参
    -> 固化 Pipeline，保存并复现
```

推荐先掌握这组基础 API：

```python
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
)
```

最小回归模板：

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(X_train, y_train)
prediction = model.predict(X_test)
```

关键约定：

- `X` 通常是二维 `(样本数, 特征数)`，`y` 回归时通常是 `(样本数,)`。
- `random_state` 用于复现实验；随机种子固定不等于结果永远完全相同。
- 测试集只在最终比较模型时使用，不能反复据此调参。
- 预处理应放进 `Pipeline`，否则交叉验证很容易产生数据泄漏。

## 4. 分阶段学习路线

### 阶段 0：先建立“问题地图”

**学习目标**：分清模型的预测对象和输出形式。

| 问题类型 | 目标 `y` | 首个基线 | 经典业务问题 |
| --- | --- | --- | --- |
| 回归 | 连续数值 | `DummyRegressor`、`LinearRegression` | 房价、销量、能耗、延迟预测 |
| 二分类 | 0/1 或两类标签 | `DummyClassifier`、`LogisticRegression` | 垃圾邮件、是否流失、是否违约 |
| 多分类 | 多个类别 | `LogisticRegression(multi_class=...)` | 文本主题、故障类型、手写数字 |
| 高维稀疏预测 | 大量 0/1 或词频特征 | `Ridge`、`LogisticRegression` | TF-IDF 文本分类、点击率特征 |

**练习**：拿一个数据集写出数据字典：每列的含义、单位、是否在预测时可获得、是否可能泄漏标签、目标的业务代价。

---

### 阶段 1：一元与多元普通最小二乘回归

#### 1.1 `LinearRegression`

**模型形式**：

\[
\hat y = w_1x_1 + w_2x_2 + \cdots + w_px_p + b
\]

通过最小化残差平方和（OLS）寻找参数。`sklearn.linear_model.LinearRegression` 默认使用最小二乘求解，不需要手写正规方程。

**经典问题**：

- 根据面积、房间数、地段等特征预测房价。
- 根据广告投入预测销售额。
- 根据温度、湿度、时间等变量预测能耗。
- 用实验变量解释一个连续结果。

**必须掌握**：

- 一元线性回归、多元线性回归和截距。
- `coef_`、`intercept_`、`predict()`、`score()`。
- MSE、RMSE、MAE、`R^2` 的含义和局限。
- 线性关系、独立同分布、残差均值为 0、同方差等基本假设。
- 用散点图、预测值-真实值图、残差图检查拟合。

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_, model.intercept_)
pred = model.predict(X_test)
```

**必须做的手算/验证**：用 NumPy 在只有一个特征的小数据集上计算斜率和截距，确认它们与 sklearn 结果一致；再比较训练集 `R^2` 与测试集 `R^2`。

#### 1.2 何时普通线性回归不够

- 特征高度相关时，系数可能不稳定，解释会失真。
- 特征数量接近或超过样本数量时，容易过拟合或矩阵病态。
- 存在强异常值时，平方损失会被少数样本主导。
- 真实关系明显是曲线时，直接拟合直线会欠拟合。
- 目标分布严重偏斜时，可能需要变换目标或换更合适的模型。

---

### 阶段 2：线性模型的评估与数据诊断

这一阶段不引入太多新模型，但决定后续结论是否可靠。

#### 2.1 回归指标

| 指标 | 关注点 | 适合场景 | 注意事项 |
| --- | --- | --- | --- |
| MAE | 平均绝对误差 | 误差单位与业务一致 | 对异常值较稳健 |
| MSE | 平方误差 | 大误差代价特别高 | 对异常值敏感 |
| RMSE | MSE 开平方 | 与目标同单位 | 仍对大误差敏感 |
| `R^2` | 相对基线的解释程度 | 比较同一目标上的模型 | 可能为负，不能替代业务误差 |
| `MAPE` | 相对百分比误差 | 目标为正且不接近 0 | `y` 接近 0 时不稳定 |

用 `mean_squared_error(..., squared=False)` 计算 RMSE 时，要注意当前 sklearn 版本的参数兼容性；也可以先计算 MSE 再开平方。

#### 2.2 交叉验证

- 回归通常使用 `KFold` 或 `RepeatedKFold`。
- 分类优先使用 `StratifiedKFold`，保持每折类别比例。
- 时间序列不能随机打乱，应使用按时间顺序的切分方法。
- 交叉验证的预处理必须在 `Pipeline` 内，不能先对全量数据 `fit_transform`。

#### 2.3 数据泄漏清单

- 用全量数据计算均值、标准差、缺失值填充值。
- 用测试集选择特征、阈值或超参数。
- 把预测时才知道的字段（如事后状态、未来金额）作为特征。
- 在同一用户、设备或订单的重复记录随机分到训练和测试两侧。
- 先做全量目标编码、特征选择，再交叉验证。

---

### 阶段 3：多项式回归与特征变换

#### 3.1 `PolynomialFeatures + LinearRegression`

**经典问题**：房价随面积不是直线增长、温度与能耗呈弯曲关系、存在特征交互作用。

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression(),
)
model.fit(X_train, y_train)
```

必须理解：多项式回归仍然是线性模型，因为它对展开后的特征参数线性；`degree` 越大，特征数量和过拟合风险通常越高。

#### 3.2 需要配套掌握的变换

- `StandardScaler`：让不同量纲特征处于可比较尺度，尤其是正则化模型必需。
- `MinMaxScaler`：把特征缩放到固定区间，注意异常值影响。
- `RobustScaler`：使用中位数和四分位距，对异常值更稳健。
- `log1p`、幂变换：处理右偏、长尾的正值特征或目标。
- 交互项：表达“某个特征的作用取决于另一个特征”的关系。

**练习**：在同一数据集上比较一次线性、二次多项式和三次多项式模型，绘制训练/验证误差曲线，说明偏差-方差变化。

---

### 阶段 4：正则化回归家族

正则化是在损失函数中加入对参数大小的惩罚，用来降低模型复杂度和系数不稳定性。正则化前通常要标准化特征。

#### 4.1 `Ridge`：L2 正则化

\[
\min_w \|y-Xw\|_2^2 + \alpha\|w\|_2^2
\]

**经典问题**：特征高度相关、特征维度较高、希望保留全部特征但降低过拟合。

- 系数通常变小但不容易精确变成 0。
- `alpha` 越大，模型越保守，偏差增大、方差通常降低。
- 多重共线性下通常比普通最小二乘更稳定。

```python
from sklearn.linear_model import Ridge

model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
```

#### 4.2 `Lasso`：L1 正则化

\[
\min_w \frac{1}{2n}\|y-Xw\|_2^2 + \alpha\|w\|_1
\]

**经典问题**：希望自动筛选特征、建立可读的稀疏模型、处理大量无关特征。

- 部分系数可以被压到 0。
- 相关特征之间可能随机保留其中一部分，解释要谨慎。
- 特征很多、样本较少时很有价值，但 `alpha` 需要交叉验证。

#### 4.3 `ElasticNet`：L1 + L2

**经典问题**：特征高度相关，同时希望保留稀疏性和稳定性，例如基因、文本或大量工程特征。

- `alpha` 控制总体正则化强度。
- `l1_ratio` 控制 L1 与 L2 的混合比例。
- `l1_ratio=1` 接近 Lasso，`l1_ratio=0` 接近 Ridge（实际使用时按 API 约束设置）。

#### 4.4 自动选择正则强度

重点掌握：`RidgeCV`、`LassoCV`、`ElasticNetCV`。它们通过交叉验证选择候选超参数，适合作为第一版基线；仍需保留独立测试集做最终评估。

**模型选择顺序**：

1. 先用 `LinearRegression` 建立无正则基线。
2. 特征相关或维度较高，优先试 `Ridge`。
3. 需要稀疏特征选择，试 `Lasso`。
4. 相关特征成组出现且仍需稀疏，试 `ElasticNet`。
5. 用交叉验证比较误差、系数稳定性和业务可解释性，而不是只看单次测试分数。

---

### 阶段 5：异常值与非典型回归

#### 5.1 `HuberRegressor`

**经典问题**：数据中有少量异常点，但不希望它们像 MSE 那样强烈影响拟合。

Huber 损失在误差较小时近似平方损失，误差较大时转为近似绝对损失。它不是“自动修复脏数据”，仍应先调查异常点的来源。

#### 5.2 `RANSACRegressor`

**经典问题**：大部分样本满足某个线性关系，但有一批明显离群点或错误观测，例如从图像/传感器中估计几何关系。

它会尝试从随机子集拟合模型并识别内点。随机性、内点比例和残差阈值需要通过验证集检查。

#### 5.3 `TheilSenRegressor`

**经典问题**：样本量不太大、异常值较多、需要一个比 OLS 更稳健的线性趋势估计。

它通常比普通最小二乘更抗异常值，但计算成本可能更高。

#### 5.4 目标分位数与不对称代价：`QuantileRegressor`

**经典问题**：不只想预测平均值，而是要预测 P50、P90 等分位数，例如库存安全量、配送时长上界、风险预警阈值。

它最小化分位数损失，适合表达“低估和高估代价不同”的业务需求。评估时要检查覆盖率和分位数损失，不能只看 MAE。

---

### 阶段 6：逻辑回归与线性分类

线性模型不只解决回归问题。分类任务中，`LogisticRegression` 学习输入特征的线性组合，再通过 sigmoid/softmax 映射为概率。

#### 6.1 `LogisticRegression`

**二分类形式**：

\[
p(y=1|x)=\sigma(w^Tx+b)=\frac{1}{1+e^{-(w^Tx+b)}}
\]

**经典问题**：

- 判断一封邮件是否为垃圾邮件。
- 预测用户是否流失、交易是否欺诈、贷款是否违约。
- 用 TF-IDF 特征做文本主题或情感分类。
- 需要输出概率并解释特征方向的业务分类。

**必须掌握**：

- `predict()` 输出类别，`predict_proba()` 输出概率。
- 默认阈值通常是 0.5，但业务上应依据代价调整。
- `C` 是正则化强度的倒数，`C` 越小正则化越强。
- `solver`、`penalty`、`max_iter` 的兼容关系和收敛告警。
- 二分类、多分类、多标签问题的区别。

```python
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, max_iter=1000),
)
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
label = model.predict(X_test)
```

#### 6.2 分类评估

| 指标 | 适合关注 |
| --- | --- |
| Accuracy | 类别较均衡且误判代价接近 |
| Precision | 误报代价高，例如人工审核资源有限 |
| Recall | 漏报代价高，例如风险/疾病筛查 |
| F1 | 希望兼顾 Precision 与 Recall |
| ROC-AUC | 比较整体排序能力，类别相对不极端时常用 |
| PR-AUC | 正类稀少时更有信息 |
| Log loss | 关注概率质量和校准 |

一定要查看混淆矩阵，并根据业务代价选择阈值；不要在严重类别不平衡时只报告 Accuracy。

---

### 阶段 7：大规模、稀疏和在线学习

#### 7.1 `SGDRegressor` 与 `SGDClassifier`

**经典问题**：数据量太大，无法每次都用全部样本求解；数据以流式方式到达；特征是高维稀疏矩阵。

- 使用随机梯度下降，适合大规模数据和在线更新。
- 对学习率、批次、正则化和特征缩放敏感。
- 需要关注收敛、`max_iter`、`tol` 和 `early_stopping`。
- 文本 TF-IDF、日志流和推荐系统特征是典型场景。

#### 7.2 `PassiveAggressiveClassifier/Regressor`

**经典问题**：在线分类、文本流或需要快速对新样本更新的场景。

它在预测正确时尽量不更新，预测错误时进行较激进的更新。适合特定在线学习任务，不是通用的离线首选。

#### 7.3 计数和概率模型：`PoissonRegressor` 等广义线性模型

**经典问题**：目标是计数或非负值，例如事故次数、呼叫量、订单数。

线性回归假设目标误差结构近似高斯，而广义线性模型通过链接函数适配不同分布。学习时要理解“线性预测子 + 链接函数 + 目标分布”的关系。

## 5. 统一比较表：遇到问题如何选模型

| 需求/症状 | 首选尝试 | 主要原因 | 需要警惕 |
| --- | --- | --- | --- |
| 想先建立透明基线 | `LinearRegression` / `LogisticRegression` | 简单、可解释、速度快 | 欠拟合、共线性 |
| 特征相关、系数不稳定 | `Ridge` | L2 收缩，保留全部特征 | 特征必须合理缩放 |
| 想得到稀疏特征 | `Lasso` | L1 可把系数压到 0 | 相关特征选择不稳定 |
| 相关特征成组且仍想稀疏 | `ElasticNet` | L1 与 L2 折中 | 两个超参数需验证 |
| 关系明显弯曲 | 多项式特征 + Ridge | 增加曲线/交互表达能力 | 特征爆炸、过拟合 |
| 少量异常值影响很大 | `HuberRegressor` | 对大残差降权 | 异常点仍需业务调查 |
| 大量错误点或只有部分内点 | `RANSACRegressor` | 显式寻找一致子集 | 随机性和阈值敏感 |
| 关注 P90/P95 风险上界 | `QuantileRegressor` | 直接拟合分位数 | 指标要用分位数损失/覆盖率 |
| 样本和特征规模很大 | `SGDRegressor/Classifier` | 增量式近似优化 | 缩放、收敛、学习率 |
| 目标是类别概率 | `LogisticRegression` | 概率输出和线性决策边界 | 阈值、校准、不平衡 |
| 目标是计数 | `PoissonRegressor` | 与计数分布更匹配 | 目标非负、分布假设 |

## 6. 必做实践项目

### 项目 A：房价回归（连续值 + 解释）

**目标**：从数据清洗到报告，比较 `LinearRegression`、`Ridge`、`Lasso` 和二次多项式模型。

要求：

- 明确训练/验证/测试划分和随机种子。
- 数值缺失值填补、类别独热编码、数值标准化全部放入 `ColumnTransformer + Pipeline`。
- 同时报告 MAE、RMSE、`R^2`，解释指标差异。
- 比较系数大小、稀疏性和交叉验证结果。
- 画预测值-真实值图和残差图，记录最差样本并解释原因。

### 项目 B：文本垃圾邮件/情感分类（高维稀疏 + 分类）

要求：

- 使用 `TfidfVectorizer + LogisticRegression`，再与 `LinearSVC` 或 `SGDClassifier` 比较。
- 报告混淆矩阵、Precision、Recall、F1、ROC-AUC 或 PR-AUC。
- 调整分类阈值，说明误报和漏报的取舍。
- 检查是否把测试集词表信息泄漏到训练过程。

### 项目 C：订单/配送时长分位数预测（业务代价不对称）

要求：

- 对比 `Ridge` 的均值预测和 `QuantileRegressor` 的 P50/P90 预测。
- 计算分位数损失与覆盖率。
- 说明“平均时长预测”和“保证大多数订单不超时”不是同一个目标。

### 项目 D：传感器数据鲁棒回归（异常值）

要求：

- 人为注入不同强度的异常值。
- 比较 `LinearRegression`、`Ridge`、`HuberRegressor` 和 `RANSACRegressor`。
- 记录异常比例变化时各模型的误差和系数变化。

## 7. 推荐学习顺序与时间安排

可以按每天约 1.5～2 小时完成，速度可按基础调整：

| 周期 | 主题 | 产出 |
| --- | --- | --- |
| 第 1 天 | 回归/分类问题、NumPy 矩阵和最小二乘直觉 | 手算一元回归，完成数据字典 |
| 第 2～3 天 | `LinearRegression`、指标、残差诊断 | 一个可复现的回归基线 |
| 第 4 天 | 切分、交叉验证、Pipeline、泄漏 | 一份无泄漏的预处理流水线 |
| 第 5 天 | 多项式特征和标准化 | 比较 degree 1/2/3 的偏差-方差 |
| 第 6～7 天 | `Ridge`、`Lasso`、`ElasticNet`、CV | 正则化模型对比表 |
| 第 8 天 | 异常值模型与分位数回归 | 鲁棒回归实验报告 |
| 第 9～10 天 | `LogisticRegression` 和分类指标 | 二分类项目及阈值分析 |
| 第 11 天 | SGD、稀疏特征和在线学习 | 大规模/文本基线 |
| 第 12 天 | 综合项目和模型选择报告 | 完整 README、代码和结论 |

## 8. 常见坑与排查顺序

1. **`ConvergenceWarning`**：先检查特征尺度、`max_iter`、正则化参数和异常值，不要只盲目增加迭代次数。
2. **训练分数高、测试分数低**：检查数据泄漏、特征数量、多项式阶数、正则化强度和切分方式。
3. **系数大小无法直接比较**：不同量纲下系数不可横向比较；先标准化，并明确解释的是标准化后的影响。
4. **`R^2` 很高但业务误差仍不可接受**：补充 MAE/RMSE，并把误差换算回业务单位。
5. **类别不平衡导致 Accuracy 虚高**：查看混淆矩阵、Recall、Precision、F1、PR-AUC。
6. **测试集分数反复变化**：固定随机种子、使用交叉验证，并减少对测试集的重复试验。
7. **特征列顺序错乱**：保存完整 `Pipeline`，不要只保存模型参数后手动拼接特征。
8. **预测时出现训练中没有的类别**：为 `OneHotEncoder` 正确配置未知类别处理，并验证线上数据分布。
9. **目标含大量 0 或负值却直接用 MAPE/对数变换**：先确认指标和变换的数学前提。
10. **把相关性当因果性**：线性模型的系数描述条件关联，不能自动证明因果关系。

## 9. 最终验收清单

完成学习路线后，应该能独立完成以下任务：

- [ ] 解释 `LinearRegression` 的损失、参数、截距和 `R^2`。
- [ ] 用 `Pipeline` 完成缺失值填补、缩放、独热编码和模型训练。
- [ ] 说明 `Ridge`、`Lasso`、`ElasticNet` 的惩罚项和使用场景。
- [ ] 用交叉验证选择正则化强度，而不是用测试集调参。
- [ ] 发现并说明多重共线性、异常值、过拟合和数据泄漏。
- [ ] 用多项式特征表达非线性，并控制特征爆炸和过拟合。
- [ ] 用 `LogisticRegression` 输出类别和概率，按业务代价选择阈值。
- [ ] 在类别不平衡时选择合适指标并解释混淆矩阵。
- [ ] 说清 `HuberRegressor`、`RANSACRegressor`、`QuantileRegressor` 和 SGD 系列分别解决什么问题。
- [ ] 完成至少一个回归项目和一个分类项目，提交代码、实验记录、指标、图表和模型选择结论。

## 10. 推荐 API 速查

```python
# 回归
from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, Lasso, LassoCV,
    ElasticNet, ElasticNetCV, HuberRegressor, RANSACRegressor,
    TheilSenRegressor, QuantileRegressor, SGDRegressor, PoissonRegressor,
)

# 分类
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, PassiveAggressiveClassifier,
)

# 特征与预处理
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    PolynomialFeatures, OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```

学习时不必一次记住全部参数。优先记住每个模型解决的**经典问题**、它的主要假设、最重要的超参数、适合的评估指标，以及如何放进无泄漏的 `Pipeline`。

## 11. 线性回归实战练习项目

下面的项目按照“先跑通，再诊断，再改进”的顺序设计。每个项目都要求保存代码、运行结果和一段结论，不要只在 Notebook 中执行单元格而不记录比较结果。

### 项目 1：广告投入预测销售额（一元线性回归）

**要解决的问题**：已知电视广告投入，预测销售额，理解斜率和截距的业务含义。

**数据**：可以使用 seaborn 的 `advertising` 数据集；如果没有该数据集，自己创建下面的数组。

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 每行是 [电视广告投入, 销售额]
data = np.array([
    [10, 12], [20, 18], [30, 25], [40, 29], [50, 36],
    [60, 42], [70, 45], [80, 53], [90, 58], [100, 65],
])
X = data[:, [0]]
y = data[:, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print("斜率:", model.coef_[0])
print("截距:", model.intercept_)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
print("广告投入 75 时的预测销售额:", model.predict([[75]])[0])
```

**练习要求**：

- 手算一条预测公式 `销售额 = w * 广告投入 + b`。
- 绘制散点图和回归直线。
- 比较训练集和测试集的 `R^2`。
- 解释“广告投入增加 1 个单位，预测销售额平均增加多少”。

### 项目 2：波士顿房价预测（多元线性回归）

**要解决的问题**：使用多个房屋和区域特征预测 `MEDV`，这是学习多元线性回归、误差分析和特征解释的基础项目。

**数据**：使用项目中的 `llm02_ml_dl/data/HousingData/HousingData.csv`。

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../../data/ml/HousingData.csv")

# 练习阶段可以先删除缺失值；正式版本改为 Pipeline + SimpleImputer
df = df.dropna()
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

result = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_,
}).sort_values("coefficient")
print(result)
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R2:", r2_score(y_test, pred))
```

**练习要求**：

- 检查每列的缺失数量、数据类型、重复行和目标值分布。
- 画出真实值与预测值散点图，观察是否存在系统性偏差。
- 找出绝对误差最大的 10 个样本，分析它们是否属于异常区域。
- 不要直接根据系数绝对值判断特征重要性，先完成标准化版本再比较。
- 使用验证集选择模型，最后只用一次测试集做最终报告。

### 项目 3：房屋价格曲线拟合（多项式回归）

**要解决的问题**：当房屋面积与价格不是简单直线关系时，使用多项式特征表达弯曲趋势。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
area = np.linspace(20, 200, 100).reshape(-1, 1)
price = 2.5 * area[:, 0] + 0.015 * area[:, 0] ** 2
price += rng.normal(0, 25, size=len(area))

X_train, X_test, y_train, y_test = train_test_split(
    area, price, test_size=0.2, random_state=42
)

for degree in [1, 2, 3, 8]:
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(degree, mean_squared_error(y_test, pred) ** 0.5)
```

**练习要求**：

- 比较 degree 为 1、2、3、8 时的训练误差和测试误差。
- 画出每个模型的拟合曲线。
- 说明为什么高阶多项式可能训练集很好、测试集很差。
- 在多项式特征后加入 `Ridge`，观察正则化是否降低过拟合。

### 项目 4：Ridge、Lasso、ElasticNet 特征选择对比

**要解决的问题**：特征很多且彼此相关时，比较普通最小二乘、L2、L1 和混合正则化。

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X, y = make_regression(
    n_samples=400, n_features=60, n_informative=10,
    noise=15, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "linear": LinearRegression(),
    "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20))),
    "lasso": make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=10000, random_state=42)),
    "elasticnet": make_pipeline(StandardScaler(), ElasticNetCV(cv=5, max_iter=10000, random_state=42)),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "MAE:", mean_absolute_error(y_test, pred))
```

**练习要求**：

- 记录每个模型的测试 MAE 和交叉验证选出的正则化参数。
- 统计 Lasso 和 ElasticNet 中接近 0 的系数数量。
- 重复改变 `random_state`，观察 Lasso 特征选择是否稳定。
- 说明为什么正则化模型必须先进行特征缩放。

### 项目 5：异常值下的鲁棒回归

**要解决的问题**：少数错误传感器读数或录入错误会严重影响普通线性回归时，比较鲁棒模型。

```python
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = np.linspace(0, 10, 150).reshape(-1, 1)
y = 3 * X[:, 0] + 5 + rng.normal(0, 1, size=len(X))

# 制造 10 个异常目标值
outlier_index = rng.choice(len(X), size=10, replace=False)
y[outlier_index] += rng.normal(30, 5, size=10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
models = {
    "linear": LinearRegression(),
    "huber": HuberRegressor(),
    "ransac": RANSACRegressor(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "MAE:", mean_absolute_error(y_test, pred))
```

**练习要求**：

- 分别注入 0%、5%、10%、20% 的异常值，记录各模型 MAE。
- 比较各模型的斜率和截距与真实关系 `y = 3x + 5` 的差异。
- 说明异常点是真实业务事件还是数据错误，不能只根据模型结果删除样本。

### 项目 6：配送时长预测与分位数回归

**要解决的问题**：平均配送时长不能回答“希望 90% 的订单在多久内送达”，使用分位数回归预测 P50 和 P90。

```python
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
distance = rng.uniform(1, 30, size=500)
traffic = rng.integers(0, 4, size=500)
noise = rng.exponential(scale=3, size=500)
duration = 8 + 1.8 * distance + 5 * traffic + noise

X = np.column_stack([distance, traffic])
X_train, X_test, y_train, y_test = train_test_split(
    X, duration, test_size=0.2, random_state=42
)

for quantile in [0.5, 0.9]:
    model = QuantileRegressor(quantile=quantile, alpha=0.01, solver="highs")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("quantile:", quantile)
    print("MAE:", mean_absolute_error(y_test, pred))
    print("系数:", model.coef_, "截距:", model.intercept_)
```

**练习要求**：

- 比较 P50 与 P90 的预测值，解释两者的业务含义。
- 计算真实时长小于 P90 预测值的比例，检查是否接近 90%。
- 说明为什么 P90 不能使用普通 MSE 作为唯一评价依据。

### 项目 7：用真实时间顺序预测每日订单量

**要解决的问题**：根据日期、促销、节假日和历史订单量预测下一天订单量，练习时间切分和防止未来信息泄漏。

```python
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("daily_orders.csv", parse_dates=["date"])
df = df.sort_values("date").copy()
df["day_of_week"] = df["date"].dt.dayofweek
df["orders_lag_1"] = df["orders"].shift(1)
df["orders_lag_7"] = df["orders"].shift(7)
df = df.dropna()

features = ["promotion", "holiday", "day_of_week", "orders_lag_1", "orders_lag_7"]
split = int(len(df) * 0.8)
X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
y_train, y_test = df["orders"].iloc[:split], df["orders"].iloc[split:]

model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
```

**练习要求**：

- 禁止随机打乱数据，必须按时间先后切分。
- 只能使用预测时已经知道的字段，不能使用当天结束后才产生的订单统计。
- 与“永远预测昨天订单量”的基线比较。
- 分析节假日或促销系数是否稳定，说明线性模型的局限。

## 12. 每个实战项目的统一提交格式

每个项目至少提交以下内容：

```text
project_name/
├── README.md              # 问题、数据、方法、指标和结论
├── train.py 或 notebook   # 可从头运行的训练代码
├── requirements.txt       # 依赖版本
├── data_sample.csv        # 可公开的小样本或数据说明
└── figures/
    ├── prediction_vs_true.png
    └── residuals.png
```

`README.md` 必须回答：

1. 预测目标是什么，预测时哪些信息可用？
2. 为什么选择线性回归，而不是直接使用树模型？
3. 训练集、验证集和测试集如何划分？是否存在数据泄漏？
4. 使用了哪些指标，指标对应什么业务含义？
5. 哪些样本预测最差，可能原因是什么？
6. 模型有哪些不能解决的问题，下一步准备如何改进？

建议完成顺序：先完成项目 1 和项目 2，再完成项目 3、4；项目 5、6、7 用来训练异常值、业务代价和时间数据场景下的建模判断。
