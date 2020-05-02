#!/usr/bin/env python
# coding: utf-8

# In[9]:


# 数据预处理

# 导入包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[10]:


# 导入数据集
dataset = pd.read_csv('数据清理.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# In[11]:


X


# In[12]:


y


# In[13]:


# 处理丢失数据
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X


# In[14]:
# 0, 44, 72000
# 1, 27, 48000
# 2, 30, 54000
# 1, 38, 61000

# 1, 0, 0, 44, 72000
# 0, 1, 0, 27, 48000
# 0, 0, 1, 30, 54000
# 0, 1, 0, 38, 61000



# 编码类别变量与自变量
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# 编码因变量
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[15]:


X


# In[16]:


y


# In[ ]:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_train

