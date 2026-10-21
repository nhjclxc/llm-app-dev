

[scikit-learn官方文档](https://scikit-learn.org/stable)


[scikit-learn linear model 官方文档](https://scikit-learn.org/stable/api/sklearn.linear_model.html)


https://github.com/scikit-learn/scikit-learn.git


https://sklearn.org/stable/supervised_learning.html

https://scikit-learn.org/stable/user_guide.html




# 一、sklearn 到底要学哪些模型？

## 第一优先级：必须掌握

建立机器学习基础，理解训练、预测、评估的完整流程。

### 线性回归 LinearRegression

连续值的预测问题，
经典问题：波士顿房价、广告销售额预测、用电量预测
- [lm01_boston-housing.ipynb](sklearn01_linear_model/lm01_boston-housing.ipynb)
- [lm02_advertising.ipynb](sklearn01_linear_model/lm02_advertising.ipynb)

### 岭回归 Ridge

预测连续数值，
与 LinearRegression 不同的是 Ridge 加入了 L2 正则化来进行优化。
Ridge 并不是用来解决另一种预测任务的，而是在线性回归的基础上，通过正则化改善模型的泛化能力和稳定性。


### 逻辑回归 LogisticRegression

分类问题
经典问题：心脏疾病预测、垃圾邮件分类
- [lm03_heart-disease.ipynb](sklearn01_linear_model/lm03_heart-disease.ipynb)
- [lm05_SMSSpamCollection.ipynb](sklearn01_linear_model/lm05_SMSSpamCollection.ipynb)


### K 近邻 KNeighborsClassifier

- KNeighborsClassifier 分类
- KNeighborsRegressor 回归

学习距离度量、邻居数量 k、特征尺度对距离的影响。

经典项目：鸢尾花分类、用户相似度分析
- [neighbors01_KNeighborsClassifier.ipynb](sklearn02_neighbors/neighbors01_KNeighborsClassifier.ipynb)
- [neighbors02_KNeighborsRegressor.ipynb](sklearn02_neighbors/neighbors02_KNeighborsRegressor.ipynb)


### 决策树 DecisionTreeClassifier

DecisionTreeClassifier、DecisionTreeRegressor

学习分类和回归、树的分裂、特征重要性、过拟合，以及集成学习的基本思想。

经典项目： 心脏疾病分类、用户流失预测、房价预测。
- [tree01_DecisionTreeClassifier.ipynb](sklearn03_tree/tree01_DecisionTreeClassifier.ipynb)


### 随机森林 RandomForestClassifier

RandomForestClassifier、RandomForestRegressor

学习分类和回归、树的分裂、特征重要性、过拟合，以及集成学习的基本思想。

经典项目： 心脏疾病分类、用户流失预测、房价预测。

### 聚类 K-Means、DBSCAN

学习无监督学习、聚类中心、密度聚类、聚类数量和聚类评价。

KMeans：K-Means 聚类算法，需要提前指定聚类数量 n_clusters。

DBSCAN：基于密度的聚类算法，不需要提前指定聚类数量，可以识别噪声点。


经典项目： 用户分群、商品分群、行为模式分析。
- 用户分群：Customer Segmentation


## 第二优先级：建议掌握

用于传统分类、异常检测、特征分析和无监督学习。
### 朴素贝叶斯 MultinomialNB

MultinomialNB、BernoulliNB、GaussianNB

学习贝叶斯公式、条件独立假设、类别概率，以及文本词频特征如何参与分类。

经典项目： 垃圾短信识别、新闻文本分类。可以和你已学的 CountVectorizer、TfidfVectorizer 结合。

### 支持向量机 SVC / LinearSVC

SVC、LinearSVC、SVR

学习分类边界、最大间隔、核函数，以及特征标准化对模型的影响。

经典项目： 文本分类、垃圾短信识别、小样本分类。

### 梯度提升树 HistGradientBoostingClassifier

HistGradientBoostingClassifier、HistGradientBoostingRegressor，以及了解 GradientBoostingClassifier。

学习多个弱模型逐步纠正误差的思路，理解 boosting 和随机森林的区别。

经典项目： 用户转化率预测、信用风险分类、结构化数据预测。

### 降维算法 主成分分析 PCA

PCA

学习高维数据降到低维的思路、方差保留和可视化。

经典项目： 高维数据探索、特征压缩、聚类前的数据分析。

### 异常检测 IsolationForest、LocalOutlierFactor

IsolationForest、LocalOutlierFactor

学习异常样本识别，以及没有完整异常标签时如何探索数据。

经典项目： 异常访问检测、异常交易筛查、服务器指标异常分析。
学习异常样本识别，以及没有完整异常标签时如何探索数据。

经典项目： 异常访问检测、异常交易筛查、服务器指标异常分析。

### 

# 三、推荐你的实际学习顺序
我建议按下面的顺序推进，而不是按照 sklearn 的模块目录逐个学习。
01

传统分类模型

DecisionTreeClassifier → RandomForestClassifier → SVC → KNeighborsClassifier → MultinomialNB

目标：理解不同算法的原理、适用场景和优缺点。



02

集成学习

RandomForestClassifier → HistGradientBoostingClassifier → VotingClassifier → StackingClassifier

目标：理解 Bagging、Boosting，以及如何组合模型。



03

无监督学习与降维

KMeans → DBSCAN → PCA → IsolationForest

目标：理解聚类、降维和异常检测。



04

模型工程化

Pipeline → ColumnTransformer → cross_val_score → GridSearchCV

目标：把数据预处理、训练、验证、调参组织成可靠的工作流。



05

传统 NLP 项目

TfidfVectorizer + MultinomialNB / LogisticRegression / LinearSVC

目标：完成垃圾短信识别、文本分类，并比较不同模型的效果。

# 

注意：sklearn 是一个机器学习工具库，不是单纯的模型集合。特征预处理、模型评估、交叉验证、超参数搜索和流水线，在实际开发中同样重要。
二、每类模型要解决什么经典问题？
这是最值得你花时间的部分。不要只记住类名，要知道每种算法解决什么问题。
学习顺序	模型	经典应用场景
1	LinearRegression	波士顿房价预测（经典历史数据集）
2	Ridge	多特征、特征相关性较强的房价预测
3	LogisticRegression	心脏疾病风险分类、垃圾邮件识别
4	KNeighborsClassifier	鸢尾花分类、相似样本分类
5	DecisionTreeClassifier	用户流失预测、贷款违约分类
6	RandomForestClassifier	用户流失预测、结构化数据分类
7	MultinomialNB	短信垃圾邮件识别
8	LinearSVC	新闻分类、文本主题分类
9	HistGradientBoostingClassifier	表格数据分类、风险预测
10	KMeans	客户分群、用户行为聚类
11	DBSCAN	地理位置聚类、异常空间数据识别
12	PCA	高维特征降维、数据探索
官方用户指南也将线性模型、SVM、朴素贝叶斯、决策树、集成学习、聚类和降维作为独立的核心学习领域。

三、比多学几个模型更重要的 sklearn 知识
面向大模型应用开发，我建议你把下面这些内容也列入必学清单。
知识点	必须掌握的 API	用途
数据集划分	train_test_split	划分训练集和测试集
缺失值处理	SimpleImputer	填充缺失数据
特征标准化	StandardScaler	缩放数值特征
类别编码	OneHotEncoder	将类别特征转换为数值
文本向量化	CountVectorizer、TfidfVectorizer	将文本转换成数值特征
模型评估	accuracy_score、classification_report、confusion_matrix	评价分类效果
交叉验证	cross_val_score	评估模型的泛化能力
超参数搜索	GridSearchCV、RandomizedSearchCV	搜索更合适的模型参数
流水线	Pipeline、ColumnTransformer	串联预处理与模型，减少数据泄漏风险
模型保存	joblib	保存和加载训练好的模型
这些能力在传统机器学习项目中十分实用，也能帮助你理解后续的检索、排序、评估和数据处理流程。
四、学到什么程度就可以进入大模型应用开发？
建议以这个标准判断，而不是以学完多少个模型判断。
- [ ] 能独立完成 LinearRegression 回归项目。
- [ ] 能独立完成 LogisticRegression 分类项目。
- [ ] 能用 TF-IDF 把文本转换成特征，并完成垃圾短信识别。
- [ ] 能比较逻辑回归、朴素贝叶斯和 SVM 的效果。
- [ ] 能使用 Pipeline 串联预处理与模型。
- [ ] 能理解准确率、精确率、召回率、F1-score 和交叉验证。
- [ ] 能使用 KMeans 完成一个基本聚类项目。
完成这些后，就可以把主要精力转向 LLM 应用开发。传统机器学习不必全部学完，更不需要在 sklearn 上花费数月时间。

接下来建议按 线性回归 → 逻辑回归 → 垃圾短信分类 → 决策树与随机森林 → 朴素贝叶斯与 SVM → 聚类与降维 → Pipeline 和模型评估 的顺序推进。

