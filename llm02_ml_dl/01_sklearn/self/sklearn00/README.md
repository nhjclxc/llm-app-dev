
sklearn 包的学习


# GridSearchCV 完整使用教程：从入门到实战

`GridSearchCV` 是 Scikit-learn 中用于自动搜索模型超参数，并通过交叉验证找到较优参数组合的工具。

你正在学习 sklearn 的机器学习模型，可以把它理解成模型训练过程中的一个自动调参助手：你先告诉它有哪些参数可以尝试、每个参数尝试哪些值，它就会自动组合这些参数，逐一训练模型、评估效果，最后告诉你哪组参数表现最好。

例如，你正在学习逻辑回归、岭回归、随机森林，或者之前接触过的 `IsolationForest`，都可以用类似的思路调参。不过，不同模型支持的参数和评分指标会有所不同。

## 一、先理解 GridSearchCV 解决什么问题

假设你正在使用逻辑回归预测心脏疾病，代码如下：

```
from sklearn.linear_model import LogisticRegressionmodel = LogisticRegression(C=1.0, max_iter=1000)model.fit(X_train, y_train)
```

这里有两个参数：

- `C=1.0`：控制正则化的强度。一般来说，C 越小，正则化越强。
- `max_iter=1000`：优化算法允许执行的最大迭代次数。

问题来了：`C` 应该设置成 `0.01`、`0.1`、`1`，还是 `10`？

你可以手动写多次训练代码，但这样比较繁琐。`GridSearchCV` 就是帮你自动完成这件事的。

### GridSearchCV 的工作流程

1\. 定义候选参数

例如 C = 0.01、0.1、1、10

2\. 生成参数组合

每组参数都需要单独训练和验证

3\. 执行交叉验证

默认 cv=5，即通常进行 5 折交叉验证

4\. 比较各组平均得分

例如比较 accuracy、F1、R² 等指标

5\. 找到最佳参数并重新训练

通过 best_params\_、best_score\_、best_estimator\_ 获取结果

需要记住：GridSearchCV 不负责替你选择所有模型，也不是直接修改模型参数。它是在你指定的候选范围内，寻找交叉验证得分最高的参数组合。

## 二、先理解三个核心概念

### 1. 什么是超参数？

模型参数和超参数不是一回事。

| 类型           | 含义                | 例子                            |
| ------------ | ----------------- | ----------------------------- |
| 模型参数         | 模型训练过程中从数据中学习到的内容 | 线性回归的权重、偏置                    |
| 超参数          | 训练前由开发者设置的配置      | 逻辑回归的 `C`、随机森林的 `max_depth`   |
| GridSearchCV | 自动比较多种超参数配置       | 找出验证效果较好的 `C` 和 `max_iter` 组合 |

例如：

```
LogisticRegression(C=0.1, max_iter=1000)
```

这里 `C` 和 `max_iter` 都是超参数。

`GridSearchCV` 可以自动搜索它们，但不会替你决定是否应该使用逻辑回归还是随机森林——模型类型通常需要你自己确定，或者另行设计模型比较实验。

### 2. 什么是交叉验证？

交叉验证用于更可靠地评估一组参数的表现，避免只凭一次训练集与验证集划分来决定参数。

以 `cv=5` 为例，假设有 1000 条训练数据：

每次取 800 条用于训练，200 条用于验证。示意图中每一折轮流担任验证集。

第 1 折

验证集：第 1 份

验证

训练

训练

训练

训练

第 2 折

验证集：第 2 份

训练

验证

训练

训练

训练

第 3 折

验证集：第 3 份

训练

训练

验证

训练

训练

第 4 折

验证集：第 4 份

训练

训练

训练

验证

训练

第 5 折

验证集：第 5 份

训练

训练

训练

训练

验证

训练数据

验证数据

比如某组参数在 5 折验证中得到：

```
第 1 折：0.91
第 2 折：0.94
第 3 折：0.92
第 4 折：0.90
第 5 折：0.93
```

平均分为：

\\[ \frac{0.91+0.94+0.92+0.90+0.93}{5}=0.92 \\]

这个 `0.92` 就是该参数组合的平均交叉验证得分。

注意：交叉验证各折的得分不是模型的最终测试成绩。最终效果还应该使用没有参与调参的测试集评估。

### 3. 什么是网格搜索？

假设设置以下候选参数：

```
param_grid = {    "C": [0.01, 0.1, 1, 10],    "max_iter": [500, 1000]}
```

它会组合出：

| 编号 | C    | max_iter |
| -- | ---- | -------- |
| 1  | 0.01 | 500      |
| 2  | 0.01 | 1000     |
| 3  | 0.1  | 500      |
| 4  | 0.1  | 1000     |
| 5  | 1    | 500      |
| 6  | 1    | 1000     |
| 7  | 10   | 500      |
| 8  | 10   | 1000     |

总共 \\(4 \times 2=8\\) 组参数。

如果每组参数执行 5 折交叉验证，就通常需要进行 \\(8 \times 5=40\\) 次模型拟合，之后还会用最佳参数在整个传入的训练集上重新拟合一次（默认 `refit=True`）。

这也是为什么参数范围设置得越大，搜索通常就越耗时。

## 三、GridSearchCV 的基本语法

先看官方 API 的常用写法。

[image](https://www.google.com/s2/favicons?domain=https://scikit-learn.org\&sz=32)

scikit-learn 1.9.1 documentation

+1



```
from sklearn.model_selection import GridSearchCVgrid_search = GridSearchCV(    estimator=model,    param_grid=param_grid,    scoring="accuracy",    cv=5,    n_jobs=-1,    verbose=1,    refit=True)grid_search.fit(X_train, y_train)
```

各个参数的含义如下。

| 参数                   | 作用               | 常见设置                           |
| -------------------- | ---------------- | ------------------------------ |
| `estimator`          | 要调参的模型或 Pipeline | `LogisticRegression()`         |
| `param_grid`         | 候选超参数及其取值        | 字典或字典列表                        |
| `scoring`            | 用什么指标评估          | `accuracy`、`f1`、`roc_auc`、`r2` |
| `cv`                 | 交叉验证的折数或划分器      | `5`、`10`                       |
| `n_jobs`             | 并行任务数量           | `-1` 表示使用所有可用处理器               |
| `verbose`            | 输出训练过程信息的详细程度    | `0`、`1`、`2`                    |
| `refit`              | 搜索完成后是否用最佳参数重新训练 | 通常使用 `True`                    |
| `return_train_score` | 是否保存每组参数的训练集得分   | 默认 `False`                     |
| `pre_dispatch`       | 并行任务预先分发的数量      | 默认 `"2*n_jobs"`                |
| `error_score`        | 某组参数训练失败时如何处理    | 默认 `np.nan`，调试时可用 `"raise"`    |

初学阶段，优先掌握前七个参数即可。

### 重点理解 scoring

如果不传 `scoring`，GridSearchCV 会使用模型自身的 `score()` 方法。例如：

- 分类器通常默认使用准确率 `accuracy`。
- 回归器通常默认使用决定系数 \\(R^2\\)。

但是，默认指标不一定适合你的业务。官方文档也建议根据任务目标选择评分标准。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



常用指标如下：

| 任务 | scoring                         | 适用场景                     |
| -- | ------------------------------- | ------------------------ |
| 分类 | `"accuracy"`                    | 类别相对均衡，关注整体预测正确率         |
| 分类 | `"precision"`                   | 更关注减少误报                  |
| 分类 | `"recall"`                      | 更关注减少漏报                  |
| 分类 | `"f1"`                          | 希望兼顾 Precision 和 Recall  |
| 分类 | `"roc_auc"`                     | 评估二分类模型的排序区分能力           |
| 回归 | `"r2"`                          | 评估模型解释目标变量变化的能力          |
| 回归 | `"neg_mean_absolute_error"`     | 关注平均绝对误差，越大越好（负误差越接近零越好） |
| 回归 | `"neg_root_mean_squared_error"` | 更惩罚较大的预测误差，越大越好          |

注意： sklearn 中的损失类评分通常使用负数形式，例如 `neg_mean_squared_error`。GridSearchCV 的统一规则是分数越高越好，所以选择负误差分数时，`-10` 比 `-20` 好，因为对应的真实误差更小。

## 四、第一个完整示例：逻辑回归自动调参

使用 sklearn 自带的乳腺癌数据集，完成一个二分类任务。这个例子不依赖外部 CSV 文件，适合直接运行学习。

### 1. 准备数据

```
from sklearn.datasets import load_breast_cancerfrom sklearn.model_selection import train_test_splitdata = load_breast_cancer()X = data.datay = data.targetX_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.2,    random_state=23,    stratify=y)
```

这里：

- `X` 是特征数据。
- `y` 是标签。
- `X_train`、`y_train` 用于模型训练和参数搜索。
- `X_test`、`y_test` 留到最后评估模型。
- `stratify=y` 尽量保持训练集和测试集的类别比例一致。

最重要的是：测试集不要参与 GridSearchCV 的参数选择。

### 2. 定义模型与参数网格

```
from sklearn.linear_model import LogisticRegressionmodel = LogisticRegression(    max_iter=5000,    random_state=23)param_grid = {    "C": [0.01, 0.1, 1, 10, 100],    "solver": ["lbfgs", "liblinear"]}
```

参数组合数为：

\\[ 5\times 2=10 \\]

这里我们暂时固定 `max_iter=5000`，只搜索 `C` 和 `solver`。这样可以避免搜索空间过大。

### 3. 创建 GridSearchCV 并训练

```
from sklearn.model_selection import GridSearchCVgrid_search = GridSearchCV(    estimator=model,    param_grid=param_grid,    scoring="accuracy",    cv=5,    n_jobs=-1,    verbose=1,    refit=True)grid_search.fit(X_train, y_train)
```

运行后，GridSearchCV 会对 10 组参数分别进行 5 折交叉验证，即通常需要 50 次交叉验证模型拟合，再用最佳参数在完整的 `X_train` 上重新训练模型。

### 4. 查看搜索结果

```
# 最佳参数组合print(grid_search.best_params_)# 最佳交叉验证平均得分print(grid_search.best_score_)# 最佳模型对象print(grid_search.best_estimator_)
```

这三个属性分别回答：

| 属性                | 回答的问题              |
| ----------------- | ------------------ |
| `best_params_`    | 哪组参数最好？            |
| `best_score_`     | 这组参数的平均交叉验证得分是多少？  |
| `best_estimator_` | 使用最佳参数重新训练后的模型是什么？ |

例如，假设结果是：

```
{    "C": 0.1,    "solver": "lbfgs"}
```

这只是示意输出，不是预先保证的运行结果。真实结果以你运行代码时得到的结果为准。

### 5. 在测试集上评估最佳模型

```
from sklearn.metrics import (    accuracy_score,    classification_report,    confusion_matrix)y_pred = grid_search.predict(X_test)print("测试集准确率：", accuracy_score(y_test, y_pred))print("混淆矩阵：")print(confusion_matrix(y_test, y_pred))print(classification_report(y_test, y_pred))
```

这里直接调用 `grid_search.predict()` 即可，因为默认 `refit=True`，搜索完成后它已经保存了重新训练好的最佳模型。

不要把 `best_score_` 当成测试集准确率。 前者是训练集内部交叉验证的平均得分，后者才是独立测试集上的评估结果。

## 五、如何查看全部参数组合的结果？

只看 `best_params_` 不够。实际学习中，你还应该检查所有参数组合的得分，了解参数改变后模型效果如何变化。

GridSearchCV 把这些结果保存在 `cv_results_` 中。

[image](https://www.google.com/s2/favicons?domain=https://scikit-learn.org\&sz=32)

scikit-learn 1.9.1 documentation

+1



### 1. 转换成 DataFrame

```
import pandas as pdresults = pd.DataFrame(grid_search.cv_results_)print(results.head())
```

输出中会包含很多列，主要关注：

| 字段                | 含义             |
| ----------------- | -------------- |
| `params`          | 当前参数组合         |
| `mean_test_score` | 各折验证得分的平均值     |
| `std_test_score`  | 各折验证得分的标准差     |
| `rank_test_score` | 按验证得分排名，1 表示最好 |
| `mean_fit_time`   | 平均训练耗时         |
| `mean_score_time` | 平均评分耗时         |

如果设置 `return_train_score=True`，还会保存 `mean_train_score` 等训练得分信息。

### 2. 按照得分排序

```
results = results.sort_values(    by="rank_test_score")print(    results[        [            "params",            "mean_test_score",            "std_test_score",            "rank_test_score"        ]    ].head(10))
```

这样可以观察得分最高的前 10 组参数。

### 3. 如何理解标准差？

例如：

| 参数组合 | 平均验证准确率 | 标准差   |
| ---- | ------- | ----- |
| A    | 0.960   | 0.020 |
| B    | 0.958   | 0.004 |

A 的平均分略高，但不同折之间的波动更大；B 的平均分略低，却更稳定。

这不代表 B 一定更好。你需要结合平均分、波动、模型复杂度和业务需求综合判断。如果两个模型得分接近，通常值得进一步检查它们的稳定性与泛化表现。

## 六、进阶用法一：在 Pipeline 中使用 GridSearchCV

这是实际项目中非常重要的一种用法。

你之前学习过 `SimpleImputer`、`StandardScaler`、`OneHotEncoder`、`ColumnTransformer` 和 `Pipeline`。这些预处理步骤可以与 GridSearchCV 组合起来，让搜索同时优化预处理配置和模型超参数。

### 1. 为什么要把预处理放进 Pipeline？

假设你先对整个数据集执行：

```
X_scaled = scaler.fit_transform(X)
```

然后再划分训练集、测试集。这样标准化时已经使用了测试集的统计信息，造成数据泄漏。

正确做法是：先划分数据，再让 Pipeline 在每次交叉验证中，仅使用当前折的训练部分学习缺失值填充值、均值、标准差等信息，然后转换验证部分。

### 2. 完整代码：标准化 + 逻辑回归 + 自动调参

```
from sklearn.datasets import load_breast_cancerfrom sklearn.model_selection import train_test_split, GridSearchCVfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerfrom sklearn.linear_model import LogisticRegressionfrom sklearn.metrics import classification_report, accuracy_score# 1. 加载数据data = load_breast_cancer()X = data.datay = data.target# 2. 划分数据X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.2,    random_state=23,    stratify=y)# 3. 定义 Pipelinepipeline = Pipeline([    ("scaler", StandardScaler()),    ("model", LogisticRegression(        max_iter=5000,        random_state=23    ))])# 4. 定义参数网格param_grid = {    "model__C": [0.01, 0.1, 1, 10, 100],    "model__solver": ["lbfgs", "liblinear"]}# 5. 创建搜索器grid_search = GridSearchCV(    estimator=pipeline,    param_grid=param_grid,    scoring="accuracy",    cv=5,    n_jobs=-1,    verbose=1)# 6. 搜索参数grid_search.fit(X_train, y_train)# 7. 查看最佳结果print("最佳参数：", grid_search.best_params_)print("最佳交叉验证准确率：", grid_search.best_score_)# 8. 测试集评估y_pred = grid_search.predict(X_test)print("测试集准确率：", accuracy_score(y_test, y_pred))print(classification_report(y_test, y_pred))
```

### 3. 为什么参数名要写成 `model__C`？

因为 Pipeline 中模型的名称是 `model`，而 `C` 是这个模型的参数。

```
pipeline = Pipeline([    ("scaler", StandardScaler()),    ("model", LogisticRegression())])
```

因此，参数的完整路径是：

```
"model__C""model__solver""model__max_iter"
```

中间的双下划线 `__` 表示访问 Pipeline 内部某个步骤的参数。

假如你将步骤名称改为 `lr`：

```
pipeline = Pipeline([    ("scaler", StandardScaler()),    ("lr", LogisticRegression())])
```

对应参数就必须改成：

```
param_grid = {    "lr__C": [0.01, 0.1, 1, 10]}
```

步骤名称与参数路径必须一致。对于多层嵌套的模型，也可以使用类似的双下划线路径。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 七、进阶用法二：搜索多个评分指标

有时你不希望只看准确率。例如，安全日志异常检测更关心恶意请求是否被漏掉，那么 `recall` 或 `f1` 可能比 `accuracy` 更有参考价值。

GridSearchCV 可以同时计算多个指标，然后指定其中一个指标作为选出最佳模型的依据。

```
grid_search = GridSearchCV(    estimator=model,    param_grid=param_grid,    scoring={        "accuracy": "accuracy",        "precision": "precision",        "recall": "recall",        "f1": "f1"    },    cv=5,    n_jobs=-1,    refit="f1",    return_train_score=True)grid_search.fit(X_train, y_train)print("最佳参数：", grid_search.best_params_)print("最佳 F1：", grid_search.best_score_)
```

这里有一个关键设置：

```
refit="f1"
```

它表示用 F1 分数来确定最佳参数组合，并用该组合在完整的 `X_train` 上重新训练模型。

在多指标搜索中，如果希望获得 `best_params_`、`best_score_` 和 `best_estimator_`，应通过 `refit` 指定用于选择最佳结果的评分指标。

[image](https://www.google.com/s2/favicons?domain=https://scikit-learn.org\&sz=32)

scikit-learn 1.9.1 documentation

+1



### 查看每组参数的多个得分

```
results = pd.DataFrame(grid_search.cv_results_)print(    results[        [            "params",            "mean_test_accuracy",            "mean_test_precision",            "mean_test_recall",            "mean_test_f1",            "rank_test_f1"        ]    ].sort_values("rank_test_f1"))
```

这样可以比较：某组参数是不是准确率很高，但召回率较低；另一组参数是不是 F1 更好。

对于类别不均衡的任务，例如恶意日志只占全部日志的 1%，单纯追求准确率很可能会产生误导。

## 八、进阶用法三：使用 GridSearchCV 调整随机森林

逻辑回归学会后，建议继续练习随机森林，因为它的超参数类型更加丰富。

```
from sklearn.ensemble import RandomForestClassifierfrom sklearn.model_selection import GridSearchCVmodel = RandomForestClassifier(    random_state=23)param_grid = {    "n_estimators": [100, 200],    "max_depth": [None, 5, 10],    "min_samples_split": [2, 5],    "min_samples_leaf": [1, 2]}grid_search = GridSearchCV(    estimator=model,    param_grid=param_grid,    scoring="f1",    cv=5,    n_jobs=-1,    verbose=1)grid_search.fit(X_train, y_train)print("最佳参数：", grid_search.best_params_)print("最佳交叉验证 F1：", grid_search.best_score_)y_pred = grid_search.predict(X_test)
```

这组网格有：

\\[ 2\times3\times2\times2=24 \\]

种参数组合。

在 5 折交叉验证下，通常要进行 \\(24\times5=120\\) 次交叉验证拟合，再加上最佳模型的重新拟合。

### 这些参数分别控制什么？

| 参数                  | 含义                 |
| ------------------- | ------------------ |
| `n_estimators`      | 随机森林中树的数量          |
| `max_depth`         | 每棵决策树允许的最大深度       |
| `min_samples_split` | 一个节点至少有多少样本才允许继续分裂 |
| `min_samples_leaf`  | 一个叶节点至少保留多少样本      |

调参不是参数越大越好。例如，限制树的深度、增加叶节点所需的最小样本数，可能降低过拟合；但限制太强也可能导致欠拟合。

## 九、进阶用法四：`param_grid` 可以写成字典列表

这是一个很实用、也容易忽略的语法。

普通字典：

```
param_grid = {    "C": [0.1, 1, 10],    "solver": ["lbfgs", "liblinear"]}
```

它会尝试 `C` 与 `solver` 的所有组合。

但是，有些参数组合不适用。例如 SVM 的 `linear` 核不需要 `gamma` 参数，而 `rbf` 核通常需要调节 `gamma`。

这时可以使用字典列表：

```
from sklearn.svm import SVCfrom sklearn.model_selection import GridSearchCVmodel = SVC()param_grid = [    {        "kernel": ["linear"],        "C": [0.1, 1, 10]    },    {        "kernel": ["rbf"],        "C": [0.1, 1, 10],        "gamma": ["scale", 0.01, 0.1]    }]grid_search = GridSearchCV(    model,    param_grid,    scoring="accuracy",    cv=5,    n_jobs=-1)grid_search.fit(X_train, y_train)print(grid_search.best_params_)
```

这里不是把两个字典混在一起生成所有组合，而是分别搜索两个网格：

- `linear` 核：3 组参数。
- `rbf` 核：\\(3\times3=9\\) 组参数。

总共 12 组参数，执行 5 折交叉验证时通常需要 60 次交叉验证拟合。

使用字典列表的核心价值：可以让不同模型配置拥有不同的参数搜索空间。&#x20;

[image](https://www.google.com/s2/favicons?domain=https://sklearn.org\&sz=32)

sklearn



## 十、进阶用法五：回归模型的 GridSearchCV

如果你学习的是线性回归、岭回归、Lasso 等模型，使用方法完全类似，只是评分指标和参数不同。

例如，用岭回归预测房价：

```
from sklearn.datasets import fetch_california_housingfrom sklearn.model_selection import train_test_split, GridSearchCVfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerfrom sklearn.linear_model import Ridgefrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_scoreimport numpy as np# 1. 加载回归数据data = fetch_california_housing()X = data.datay = data.target# 2. 划分数据X_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.2,    random_state=23)# 3. 建立预处理 + 回归模型 Pipelinepipeline = Pipeline([    ("scaler", StandardScaler()),    ("ridge", Ridge())])# 4. 定义待搜索参数param_grid = {    "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}# 5. 创建搜索器grid_search = GridSearchCV(    estimator=pipeline,    param_grid=param_grid,    scoring="neg_mean_squared_error",    cv=5,    n_jobs=-1)# 6. 训练并搜索grid_search.fit(X_train, y_train)print("最佳参数：", grid_search.best_params_)print("最佳交叉验证 MSE：", -grid_search.best_score_)# 7. 测试集评估y_pred = grid_search.predict(X_test)print("MAE：", mean_absolute_error(y_test, y_pred))print("MSE：", mean_squared_error(y_test, y_pred))print("RMSE：", np.sqrt(mean_squared_error(y_test, y_pred)))print("R²：", r2_score(y_test, y_pred))
```

这里需要特别注意：

```
scoring="neg_mean_squared_error"
```

GridSearchCV 选择的是负 MSE 最大的参数组合。因此，显示实际 MSE 时需要加负号：

```
-best_score_
```

而在测试集上计算 `mean_squared_error()` 时，返回的就是正常的非负 MSE，不需要加负号。

另外，`fetch_california_housing()` 在首次使用时可能需要下载数据。如果当前环境无法联网，可以换成你已经保存的房价 CSV 数据。

## 十一、如何用 GridSearchCV 调整你正在学习的 IsolationForest？

结合你之前的访问日志异常检测项目，这里有一个重要区别：`IsolationForest` 属于无监督异常检测模型，通常调用 `fit(X_train)`，不需要 `y_train`。而 GridSearchCV 的默认评分机制不一定适用于它，因为它不像常见分类器那样直接用准确率评分。

如果你的访问日志有真实标签，例如 `benign` 和 `malicious`，可以把异常检测输出映射为这两个标签，再通过自定义评分函数比较参数。这里的标签只用于评估参数，不用于 IsolationForest 的模型拟合。

### 1. 使用真实标签评估异常检测参数

假设你已经有：

- `X_train`：用于训练的特征。
- `X_test`：用于验证的特征。
- `y_test`：对应的真实标签，取值为 `"benign"`、`"malicious"`。

可以定义一个自定义评分函数：

```
import numpy as npfrom sklearn.ensemble import IsolationForestfrom sklearn.model_selection import GridSearchCVfrom sklearn.metrics import f1_score, make_scorerdef anomaly_f1_scorer(estimator, X, y):    # IsolationForest: 1 表示正常，-1 表示异常    raw_pred = estimator.predict(X)    # 将异常检测结果映射成业务标签    pred = np.where(raw_pred == -1, "malicious", "benign")    # 以 malicious 为正类计算 F1    return f1_score(        y,        pred,        pos_label="malicious",        zero_division=0    )
```

然后搜索参数：

```
model = IsolationForest(    random_state=23,    n_jobs=1)param_grid = {    "n_estimators": [100, 200],    "max_samples": ["auto", 256],    "contamination": [0.05, 0.1, 0.15, 0.2]}grid_search = GridSearchCV(    estimator=model,    param_grid=param_grid,    scoring=anomaly_f1_scorer,    cv=5,    n_jobs=-1,    refit=True,    verbose=1)# 无监督训练：只传入 X_traingrid_search.fit(X_train)print("最佳参数：", grid_search.best_params_)print("最佳交叉验证 F1：", grid_search.best_score_)
```

这里有三个关键点：

1. `fit(X_train)` 没有传 `y_train`，因为 IsolationForest 不需要真实标签来训练。
2. `scoring` 函数接收交叉验证当前折的数据和真实标签，使用真实标签计算 F1。
3. `contamination` 会影响异常判定阈值，因此可以作为候选超参数。

但是，这段代码有一个适用前提： `X_train` 中每条样本都有对应的真实标签可供评分，且传入 GridSearchCV 的 `y` 与 `X` 的行顺序一致。GridSearchCV 会把 `y_train` 切分给评分器，但不会把它作为 IsolationForest 的训练标签。

如果你已经通过独立的训练集、验证集、测试集来划分数据，那么更推荐用训练集拟合 IsolationForest、用验证集标签比较 `contamination`，最后只用测试集评估一次。这样更容易控制参数选择过程中的数据泄漏。

另外，异常检测经常涉及类别不平衡。除了 F1，建议同时查看恶意请求的 Recall、Precision，以及正常请求的误报率，不要仅根据准确率选择模型。

## 十二、如何设置交叉验证 cv？

`cv` 不只是控制数字，它还决定数据如何划分。

| 设置                | 含义           | 适用场景                 |
| ----------------- | ------------ | -------------------- |
| `cv=3`            | 3 折交叉验证      | 计算资源有限时              |
| `cv=5`            | 5 折交叉验证      | 常用默认选择               |
| `cv=10`           | 10 折交叉验证     | 希望获得更多折的评估结果，但计算成本更高 |
| `StratifiedKFold` | 尽量保持每折类别比例一致 | 分类任务，特别是类别不均衡        |
| `KFold`           | 普通 K 折划分     | 常见回归任务               |
| `TimeSeriesSplit` | 按时间先后划分训练和验证 | 时间序列、按时间预测           |

对于分类器，`cv=5` 通常会使用分层交叉验证；对于普通回归器，通常使用 KFold。整数形式的默认划分一般不自动打乱样本顺序。

[image](https://www.google.com/s2/favicons?domain=https://scikit-learn.org\&sz=32)

scikit-learn 1.9.1 documentation

+1



如果你的访问日志是按时间产生的，例如用前几天的日志训练、预测后一天的日志，那么随机打乱数据可能导致评估结果不符合真实线上场景。此时可以考虑：

```
from sklearn.model_selection import TimeSeriesSplitcv = TimeSeriesSplit(n_splits=5)
```

再传入：

```
grid_search = GridSearchCV(    estimator=model,    param_grid=param_grid,    cv=cv,    scoring="accuracy")
```

不过，对于具有用户、会话或主机分组的数据，还要考虑是否需要按用户、会话或主机分组切分，防止同一实体的信息同时出现在训练折和验证折中。

## 十三、常见问题与错误排查

### 问题 1：参数名称写错

例如：

```
param_grid = {    "c": [0.1, 1, 10]}
```

逻辑回归的参数名是大写 `C`，不是小写 `c`。

可以先查看模型有哪些参数：

```
model.get_params()
```

对于 Pipeline：

```
pipeline.get_params().keys()
```

然后根据实际参数名称配置搜索空间。

### 问题 2：搜索时间特别长

如果有 100 组参数组合、使用 10 折交叉验证，那么通常需要 1000 次交叉验证模型拟合，外加最佳模型的重新拟合。

解决思路：

- 减少候选参数数量。
- 先搜索影响较大的参数。
- 合理设置 `n_jobs=-1`，但注意内存占用。
- 大搜索空间可以考虑 `RandomizedSearchCV`。

### 问题 3：得分很高，实际预测却不好

可能原因包括：

- 测试集参与了参数选择。
- 特征中存在目标泄漏。
- 训练集与测试集分布不同。
- 评分指标不符合业务目标。
- 数据划分方式不符合真实应用场景。

例如，在恶意请求识别中，如果恶意样本占比很低，即使模型把所有请求都预测为正常，也可能获得很高的准确率，但完全无法发现恶意请求。

### 问题 4：`best_score_` 不存在

检查是否设置了：

```
refit=False
```

当 `refit=False` 时，GridSearchCV 不会重新拟合最佳模型，`best_estimator_` 和常规的 `best_score_` 等属性也可能不可用；多指标评分下还需要检查 `refit` 的配置。

### 问题 5：出现 FitFailedWarning 或 NaN 得分

某些参数组合可能不合法，或者在某个交叉验证折中训练失败。可以在调试时设置：

```
GridSearchCV(    estimator=model,    param_grid=param_grid,    error_score="raise")
```

这样会在训练失败时直接抛出异常，更容易定位错误。排查完成后，再决定是否让失败的候选参数被记为 NaN。

## 十四、GridSearchCV 和 RandomizedSearchCV 有什么区别？

两者都可以用交叉验证来寻找较好的超参数组合，但搜索方式不同。

[image](https://www.google.com/s2/favicons?domain=https://sklearn.org\&sz=32)

sklearn



| 对比项        | GridSearchCV     | RandomizedSearchCV |
| ---------- | ---------------- | ------------------ |
| 搜索方式       | 遍历指定参数的所有组合      | 随机抽取指定数量的组合        |
| 是否遍历全部候选组合 | 是                | 不一定                |
| 计算成本       | 参数组合多时可能很高       | 可通过 `n_iter` 控制预算  |
| 适用情况       | 搜索空间小、候选值明确      | 搜索空间大、部分参数范围广      |
| 结果         | 在给定网格和评分规则下找最佳组合 | 在抽样到的组合中找最佳组合      |

如果你只有 10～30 组候选参数，GridSearchCV 通常很适合。如果参数很多、范围很广，可以先用 RandomizedSearchCV 粗略探索，再围绕表现较好的范围做更细致的 GridSearchCV 搜索。

## 十五、推荐你掌握的实战标准流程

## 机器学习调参的标准流程

1. 准备数据

   检查缺失值、标签分布、特征含义，并确认没有把目标变量或未来信息混入特征。
2. 先划分训练集和测试集

   测试集在选参数的过程中保持独立。
3. 定义模型和参数网格

   从少量关键参数开始，不要一开始就把所有参数都放入网格。
4. 选择合适的评分指标和交叉验证策略

   分类关注 Accuracy、Precision、Recall、F1 等；回归关注 MAE、RMSE、R² 等。
5. 执行 GridSearchCV

   调用 `fit()`，让搜索器比较参数组合。
6. 分析 `cv_results_`

   查看平均分、标准差、排名和耗时，不只看最佳分数。
7. 使用独立测试集评估

   调用 `predict()`，计算测试集指标，并与调参前的基线模型比较。
8. 保存模型并记录参数

   保存最佳 Pipeline 或模型，以及特征处理逻辑、评分指标和数据版本。

如果你后续希望进一步学习，可以重点掌握三个方向：Pipeline 嵌套参数搜索、多指标评估，以及 GridSearchCV 与 RandomizedSearchCV 的对比实验。

## 十六、最后总结：最需要记住的代码模板

如果你暂时不想记住所有细节，只要掌握下面这套模板，就可以用于大多数有监督分类任务。

```
from sklearn.model_selection import GridSearchCVfrom sklearn.metrics import classification_report, accuracy_score# 1. 准备模型和参数候选范围model = ...param_grid = {    "参数名1": [候选值1, 候选值2],    "参数名2": [候选值3, 候选值4]}# 2. 创建搜索器grid_search = GridSearchCV(    estimator=model,    param_grid=param_grid,    scoring="f1",    cv=5,    n_jobs=-1,    refit=True)# 3. 只使用训练集搜索参数grid_search.fit(X_train, y_train)# 4. 查看最佳参数和交叉验证得分print(grid_search.best_params_)print(grid_search.best_score_)# 5. 用最佳模型预测独立测试集y_pred = grid_search.predict(X_test)# 6. 评估最终结果print("测试集准确率：", accuracy_score(y_test, y_pred))print(classification_report(y_test, y_pred))# 7. 查看所有参数组合
```

这里的 `...` 需要替换成具体模型；如果使用 Pipeline，参数名称就要写成 `步骤名__参数名`。

官方文档可以继续参考：

- [GridSearchCV API 参考](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?utm_source=chatgpt.com)
- [超参数调优完整指南](https://scikit-learn.org/stable/modules/grid_search.html?utm_source=chatgpt.com)

学习建议： 你现在可以先用逻辑回归完成一次完整的 GridSearchCV 实验，再把相同流程迁移到岭回归、随机森林和 IsolationForest。重点不是背参数，而是理解“候选参数 → 交叉验证 → 评分比较 → 最佳模型 → 独立测试集评估”这一整套流程。