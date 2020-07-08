import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

print(x_valid.shape,y_valid.shape)
print(x_train.shape,y_train.shape)
print(x_test[0].shape,y_test.shape)

class logitRegression:
    def __int__(self,lr=0.0001):
        self.lr =lr
        self.w = None
        self.b = None

    def _sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def fit(self,x,y,iter=2000):
        self.w = 0.001 * np.random.rand(28*28).reshape((28*28,1))
        self.b = 0
        for i in range(iter):
            y_hat = self._sigmoid(np.dot(x,self.w)+self.b)
            dW = x.T.dot(y_hat - y) / x.shape(0)
            db = np.sum(y_hat - y) / x.shape(0)

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X_test):
        return np.squeeze(np.where(self.__sigmoid(np.dot(X_test, self.W) + self.b) > self.threshold, 1, 0))




