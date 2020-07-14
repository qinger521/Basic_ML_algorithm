'''
    最小二乘法实现线性回归
'''
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
data = pd.read_csv(r"boston.csv")
print(data)

# 查看数据集信息，是否有缺失值等
print(data.info())

class LinearRegression:
    '''
        基于python实现最小二乘法的线性回归
    '''

    def fit(self,X,y):
        '''
        :param X: 类数组类型 [样本数量，特征数量]
        :param y: 类数组类型 [样本数量]
        :return:
        '''
        X = np.asmatrix(X.copy()) # 保证X是一个完整的数组，防止通过切片形成的不完整的数据对象
        y = np.asmatrix(y).reshape(-1,1)
        # 通过最小二乘公式，求解最佳参数  .I 求逆  .T 求转秩
        self.w_ = (X.T * X).I * X.T * y
        print("w的shape:")
        print(self.w_.shape)
    def predict(self,X):
        '''
        :param X: 类数组类型 [样本数量，特征数量]
        :return: result 数组类型
        '''
        X = np.asmatrix(X.copy())
        result = X * self.w_
        result = np.asarray(result).ravel()
        return result

t = data.sample(len(data),random_state=0)
# 为数据增加一列
# t["intercept"] = 1 但会在最后一列
new_columns = t.columns.insert(0,"Intercept")
t = t.reindex(columns = new_columns,fill_value=1)
# t["intercept"] = 1

train_X = t.iloc[:400,:-1]
train_y = t.iloc[:400,-1]
test_X = t.iloc[400:,:-1]
test_y = t.iloc[400:,-1]

lr = LinearRegression()
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