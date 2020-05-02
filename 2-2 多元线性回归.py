#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 多元线性回归

# 导入包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# 导入数据集
dataset = pd.read_csv('公司资料.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# In[3]:


# 类别变量编码
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


# In[4]:


# 防止虚拟变量干扰
X = X[:, 1:]


# In[5]:


# 将数据集分成训练集与测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[6]:


# 尝试在训练集上运用多元线性回归训练模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[7]:


# 运用在测试集上
y_pred = regressor.predict(X_test)


# In[8]:


y_pred


# In[ ]:





