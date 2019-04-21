**k-近邻算法：**    
采用测量不同特征值之间的距离的方法进行分类

**k-近邻算法工作原理：**    
存在一个含标签的训练样本集；输入无标签的新数据后，计算新数据与训练样本集之间的距离;
取训练样本数据中前k(一般k<=20)个最相似的数据;选择k个最相似数据中出现次数最多的类别，作为新数据的类别。

**k-近邻算法优缺点：**    
优点：精度高、对异常值不敏感、无数据输入假设
缺点：计算复杂度高，空间复杂度高
适用数据范围：数值型和标称型


来源：机器学习实战（第二章）
https://github.com/pbharrin/machinelearninginaction