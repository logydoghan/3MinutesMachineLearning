# 导入包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 导入数据集
dataset = pd.read_csv('银行客户留存.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# 编码类别向量
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# 将数据集分成训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# 导入 Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# 初始化神经网络
classifier = Sequential()

# 加入输入层和第一隐藏层
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# 加入第二隐藏层
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# 加入输出层
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# 设置损失函数
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# 训练神经网络
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# 预测测试集
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# 使用混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)