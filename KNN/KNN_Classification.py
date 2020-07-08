import numpy as np
import pandas as pd

# 读取鸢尾花数据集
data = pd.read_csv(r"iris.arff.csv") # 有一个参数header 表示第几行是标题行，如果数据集没有标题行则为None
print(data)

# data.head() 表示显示前几行 默认5行 , 也可指定显示几行
# data.tail() 表示显示末尾几行 默认5行 , 也可指定显示几行
# data.sample() 表示随机抽取样本 默认1个，也可指定抽取几条

# 对class的文本值进行映射,文本->数字
data["class"] = data["class"].map({"Iris-virginica":0,"Iris-setosa":1,"Iris-versicolor":2})

# 删除数据集中的一列 data.drop("id",axis=1,inplace=True) 或者 data = data.drop("id",axis=1)

# 去除重复的数据
data.drop_duplicates(inplace=True)

# 查看各个类别有多少记录
data["class"].value_counts()
print(data)

class KNN:
    '''
        使用python实现K近邻算法。（实现分类）
    '''

    def __init__(self,k):
        '''
        :param k: int 邻居的个数
        :return:
        '''
        self.k = k

    def fit(self,X,y):
        '''
        KNN为惰性学习算法，他将数据集先保存起来，在新的数据到达时在进行判断与分类
        :param X: 类数组类型 [样本数量，属性数目]
        :param y: 类数组类型 [样本数量]
        :return:
        '''

        # 不能直接保存X，因为X的类型不统一，在此将其直接转换为np.array数组的形式
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self,X):
        '''
        根据参数传递的样本，对样本进行预测
        :param X: 类数组类型 [样本数量，属性数目]
        :return: 数组类型 预测结果
        '''
        X = np.asarray(X)
        result = []
        # 对ndarray数组进行遍历，每次取数组中的一行
        for x in X:
            dis = np.sqrt(np.sum((x - self.X) ** 2,axis=1)) # 欧式距离
            # 返回排序前在序列中的位置，因为要返回的是根据距离评出的k个y
            index = dis.argsort()
            # 取距离最近的k个元素的索引
            index = index[:self.k]
            # 返回每个元素出现的次数，元素必须是非负整数
            count = np.bincount(self.y[index])
            # 返回值最大的元素的索引
            result.append(count.argmax())
        return np.asarray(result)

# 处理数据集 但直接采样可能会导致数据的"扎堆"，所以将各个类别提取出来分别采样
t0 = data[data["class"] == 0]
t1 = data[data["class"] == 1]
t2 = data[data["class"] == 2]

# 打乱顺序  random_state 随机种子 使得实验可还原
t0 = t0.sample(len(t0),random_state=0)
t1 = t1.sample(len(t1),random_state=0)
t2 = t2.sample(len(t2),random_state=0)

# 构建训练集与测试集
train_X = pd.concat([t0.iloc[:40,:-1],t1.iloc[:40,:-1],t2.iloc[:40,:-1]],axis=0)
train_y = pd.concat([t0.iloc[:40,-1],t1.iloc[:40,-1],t2.iloc[:40,-1]],axis=0)
test_X = pd.concat([t0.iloc[40:,:-1],t1.iloc[40:,:-1],t2.iloc[40:,:-1]],axis=0)
test_y = pd.concat([t0.iloc[40:,-1],t1.iloc[40:,-1],t2.iloc[40:,-1]],axis=0)

# 创建对象，进行训练
knn = KNN(3)
knn.fit(train_X,train_y)
# 进行测试
result = knn.predict(test_X)
print("预测结果：")
print(result)
print("真实结果：")
print(np.asarray(test_y))
print("准确率：")
print(np.sum(result == test_y)/len(result))