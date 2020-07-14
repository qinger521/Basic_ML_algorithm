import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

'''
    使用KNN解决回归问题，使用鸢尾花的前三个属性，预测最后一个属性
'''

data = pd.read_csv(r"iris.arff.csv")
# 删除不需要的列
data = data.drop(["class"],axis=1)
print(data)
# 删除重复行
data.drop_duplicates(inplace=True)
print(data)

# 使用KNN解决回归问题，使用鸢尾花的前三个属性，预测最后一个属性
class KNN:
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
        for x in X:
            '''计算距离'''
            dis = np.sqrt(np.sum((x - X) ** 2,axis=1))
            # 排序后，每个元素，在原数组的索引
            index = dis.argsort()
            index = index[:self.k]
            result.append(np.mean(self.y[index]))
            # result.append(np.mean(np.multiply(self.y[index],1/dis[index])))
        return result

# 处理数据集
t = data.sample(len(data),random_state=0)
train_X = t.iloc[:120,:-1]
train_y = t.iloc[:120,-1]
test_X = t.iloc[120:,:-1]
test_y = t.iloc[120:,-1]

knn = KNN(7)
knn.fit(X=train_X,y=train_y)
result = knn.predict(test_X)
print("预测输出：")
print(result)

print("真实输出：")
print(np.asarray(test_y))

print("误差：")
print(np.sum(result-test_y) ** 2)

# 调整画布
plt.figure(figsize=(10,10))
plt.plot(result,"ro-",label="result")
plt.plot(test_y.values,"go-",label="true")
plt.show()
