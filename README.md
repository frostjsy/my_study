# my_study
  编程学习计划  
  
**计划1：《机器学习实战》**(5个月)  
第2章 k-近邻    
第3章 决策树    
第4章 朴素贝叶斯    
第5章 logistic回归    
第6章 SVM    
第7章 AdaBoost  
第8章 数值型线性回归  
第9章 树回归  
第10章 k-均值聚类算法  
第11章 Apriori算法关联分析  
第12章 FP-growth寻找频繁项集  
第13章 PCA降维  
第14章 SVD奇异矩阵分解  

**计划2：自然语言处理**（6个月）    
TF-IDF算法      
skip-gram模型      
CBOW模型      
Word2Vec模型      
Doc2Vec模型      
RNN模型    
LSTM模型    
Seq2Seq模型    
HMM模型    
CRF模型    
贝叶斯网络    

**计划3：评价指标计算**（2个月）    
TP：1-1    
FN：0-1    
FP：1-0  
TN：0-0  
Precision(P)精确度，查准率：预测对的正例/预测为正例总数 TP/TP+FP  
Recall(R)召回率，查全率：预测对的正例数/正例总数 TP/TP+FN  
sensitivity(sn)敏感度：预测对的正例数/正例总数 TP/TP+FN  
specificity(sp)表示特异性: 预测对的负例/总负例数 TN / FP + TN  
TPR表示正阳率：TP/TP+FN  
FPR表示假阳率: FP/FP+TN  
Accuracy表示准确度，正确分类率： TP+TN/TP+FN+FP+TN  
F1 表示 precision和recall的调和平均，通常用于二分类任务, 分割任务的模型评价指标：F1=2*P*R/(P+R) = 2/(1/P+1/R)  
AP：横轴R，纵轴P，曲线下面积表示AP，曲线为递减趋势  
MAP：对于多分类任务，每一类都有个AP，取各个AP的平均，即 MAP。  
ROC：横轴FPR（假阳率），纵轴TPR（正阳率），曲线通常是在y=x上方，越趋近于左上角，模型越好。  
AUC： ROC下的面积。  

**计划4：数据处理**（1个月）  
归一化   
标准化   
离散化  
