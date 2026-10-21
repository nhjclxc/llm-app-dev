

[scikit-learn官方文档](https://scikit-learn.org/stable)


[scikit-learn linear model 官方文档](https://scikit-learn.org/stable/api/sklearn.linear_model.html)


https://github.com/scikit-learn/scikit-learn.git


https://sklearn.org/stable/supervised_learning.html

https://scikit-learn.org/stable/user_guide.html




一、sklearn 到底要学哪些模型？
我把它们分成三个优先级。

第一优先级：必须掌握
建立机器学习基础，理解训练、预测、评估的完整流程。
- 线性回归 LinearRegression
- 岭回归 Ridge
- 逻辑回归 LogisticRegression
- K 近邻 KNeighborsClassifier
- 决策树 DecisionTreeClassifier
- 随机森林 RandomForestClassifier
- K-Means 聚类 KMeans
- TF-IDF 文本特征提取 TfidfVectorizer（不是模型，但对文本任务很重要）




第二优先级：建议掌握
用于传统分类、异常检测、特征分析和无监督学习。
- 朴素贝叶斯 MultinomialNB
- 支持向量机 SVC / LinearSVC
- 梯度提升树 HistGradientBoostingClassifier
- DBSCAN 聚类 DBSCAN
- 主成分分析 PCA
- 孤立森林 IsolationForest




第三优先级：了解即可
理解原理、适用场景即可，不必一开始深入研究。
- Lasso 回归 Lasso
- 弹性网络 ElasticNet
- 高斯过程 GaussianProcessRegressor
- 谱聚类 SpectralClustering
- 各类复杂集成算法及特殊估计器



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

