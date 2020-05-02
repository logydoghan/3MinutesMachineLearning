#!/usr/bin/env python
# coding: utf-8

# In[11]:


# 简单的线性回归

# 导入包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[12]:


# 导入数据集
dataset = pd.read_csv('工资数据.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# In[13]:


# 将数据集分成训练集与测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# In[14]:


# 尝试在训练集上运用简单线性回归训练模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[15]:


# 应用在测试集上
y_pred = regressor.predict(X_test)


# In[18]:


# 将训练集结果可视化
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience  (Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# In[19]:


# 将测试集结果可视化
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:





