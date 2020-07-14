import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv(r"boston.csv")
print(data)

class LinearRegression:
    '''
        使用梯度下降实现线性回归
    '''
    def __init__(self,alpha,times):
        '''
        :param alpha: 学习率 float
        :param times: 迭代次数 int
        '''
        self.alpha = alpha
        self.times = times

    def fit(self,X,y):
        '''
        :param X: 类数组类型 [样本数量，属性数量]
        :param y: 类数组类型 [样本数量]
        :return:
        '''

        X = np.asarray(X)
        y = np.asarray(y)

        # 初始化权重向量->此处初始化为0
        self.w = np.zeros(X.shape[1]+1)

        # 损失值列表 损失值=[预测值-真是值]^2求和除以2
        self.loss = []

        for i in range(self.times):
            y_hat = X.dot(self.w[1:]) + self.w[0]
            error = y - y_hat
            # 计算损失值并添加到损失列表
            self.loss.append(np.sum(error ** 2)/2)
            self.w[0] += self.alpha * np.sum(error)
            self.w[1:] += self.alpha * np.dot(X.T,error)

    def predict(self,X):
        '''
        :param X: 类数组类型 [样本数量，属性数量]
        :return: result 预测结果
        '''

        X = np.asarray(X)
        result = np.dot(X,self.w[1:]) + self.w[0]
        return result

class StandardScaler:
    '''
        对数据进行标准化处理 均值为0 标准差为1
    '''
    def fit(self,X):
        '''
        根据传递的样本每一列计算均值与标准差
        :param X: 类数组
        :return:
        '''
        X = np.asarray(X)
        # 按照每一列求标准差、均值
        self.std_ = np.std(X,axis=0)
        self.mean = np.mean(X,axis=0)

    def tranform(self,X):
        '''
        对给定的X，进行标准化处理，将X的每一列，转化为标准正态分布的形式
        :param X:
        :return:
        '''
        return (X - self.mean)/self.std_
    def fit_tranform(self,X):
        self.fit(X)
        return self.tranform(X)

t = data.sample(len(data),random_state=0)
stdX = StandardScaler()
stdY = StandardScaler()

train_X = t.iloc[:400,:-1]
train_X = stdX.fit_tranform(train_X)
train_y = t.iloc[:400,-1]
train_y = stdY.fit_tranform(train_y)
test_X = t.iloc[400:,:-1]
test_X = stdX.tranform(test_X)
test_y = t.iloc[400:,-1]
test_y = stdY.tranform(test_y)



lr = LinearRegression(0.000001,9000000)
lr.fit(train_X,train_y)
result = lr.predict(test_X)
print("预测结果为：")
print(result)
print("真实结果为：")
print(np.asarray(test_y))
print("测试误差为：")
print(np.mean(result-test_y) ** 2)

plt.figure(figsize=(10,10))
plt.plot(result,"ro-",label="predict")
plt.plot(np.asarray(test_y),"bo-",label="true")
plt.show()