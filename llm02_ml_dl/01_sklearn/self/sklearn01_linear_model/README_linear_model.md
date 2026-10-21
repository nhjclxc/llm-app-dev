

# `sklearn.linear_model` 模型

## 回归模型

预测连续数值

- LinearRegression
- Ridge
- Lasso
- ElasticNet
- SGDRegressor
- BayesianRidge
- HuberRegressor
- PoissonRegressor

## 分类模型

预测类别或类别概率

- LogisticRegression
- SGDClassifier
- Perceptron
- PassiveAggressiveClassifier

## 其他相关模型

- QuantileRegressor：预测指定分位数。
- RANSACRegressor、TheilSenRegressor：对异常值更稳健的回归方法。
- MultiTaskLasso、MultiTaskElasticNet：多个相关回归目标的联合建模。
- TweedieRegressor、GammaRegressor：针对特定目标分布的广义线性模型。

## 经典问题与解决模型
经典问题	        数据集示例	                                    任务类型	        建议学习模型
房价预测	        Boston Housing、California Housing	            回归	            LinearRegression、Ridge、Lasso
广告投入预测销售额	Advertising	                                    回归	            LinearRegression、Ridge、Lasso
糖尿病病情进展预测	Diabetes（sklearn 内置数据集）	                回归	            LinearRegression、Ridge、Lasso
心脏疾病分类	    Heart Disease	                    二分类或多分类，取决于标签定义	LogisticRegression、RidgeClassifier
垃圾邮件识别	    Spam / SMS Spam Collection	                    二分类	        LogisticRegression、SGDClassifier、Perceptron
客户流失预测	    Telco Customer Churn	                        二分类	        LogisticRegression、SGDClassifier
保险赔付金额预测	保险理赔数据	                         正值回归，部分数据也包含零值	GammaRegressor、TweedieRegressor
每日订单量预测	门店订单计数数据	                               计数回归	        PoissonRegressor
含异常点的价格预测	带有异常成交记录的房价数据	                        稳健回归	        HuberRegressor、RANSACRegressor
高维特征筛选	    大量广告渠道、用户行为特征	                        回归或分类	    Lasso、ElasticNet，或带 L1 正则化的分类器


# 评估指标

所有的评估指标函数都是使用 `sklearn.metrics` 包下提供的方法

## 线性回归：预测连续数值

典型任务：Boston Housing 房价预测。

指标	    sklearn API	            作用	                        趋势
MAE	    mean_absolute_error	    平均绝对误差	                越低越好
MSE	    mean_squared_error	    均方误差	                    越低越好
RMSE	root_mean_squared_error	均方根误差	                越低越好
R²	    r2_score	            决定系数，衡量相对基准的拟合表现	越高越好

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
pred_y = model.predict(test_X)

mae = mean_absolute_error(test_y, pred_y)
mse = mean_squared_error(test_y, pred_y)
rmse = root_mean_squared_error(test_y, pred_y)
r2 = r2_score(test_y, pred_y)

```
## 逻辑回归：预测离散类别

典型任务：heart-disease 心脏病预测。

指标	        sklearn API	        作用	                                    趋势
Accuracy	accuracy_score	    【准确率】预测正确的比例	                越高越好
Precision	precision_score	    【精确率】预测为正类的样本中，真正为正类的比例	越高越好
Recall	    recall_score	    【召回率】实际正类中，被模型识别出来的比例	    越高越好
F1	        f1_score	        【F1 分数】Precision 和 Recall 的调和平均	越高越好
ROC-AUC	    roc_auc_score	    【ROC 曲线下面积】衡量模型区分正负类别的能力	越高越好
Log Loss	log_loss	        【对数损失】衡量预测概率的质量	            越低越好
混淆矩阵	    confusion_matrix	【混淆矩阵】查看正确和错误分类的数量	        结合具体情况判断

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
# 预测类别
pred_y = model.predict(test_X)

# 预测正类的概率
pred_proba = model.predict_proba(test_X)[:, 1]

accuracy = accuracy_score(test_y, pred_y)
precision = precision_score(test_y, pred_y, zero_division=0)
recall = recall_score(test_y, pred_y, zero_division=0)
f1 = f1_score(test_y, pred_y, zero_division=0)
auc = roc_auc_score(test_y, pred_proba)
loss = log_loss(test_y, model.predict_proba(test_X), labels=[0, 1])

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("ROC-AUC:", auc)
print("Log Loss:", loss)
print("混淆矩阵:\n", confusion_matrix(test_y, pred_y))
print(classification_report(test_y, pred_y, zero_division=0))

```
## 


advertising



