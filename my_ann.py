#importing libreiries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# **part1: preparing data 
dataset= pd.read_csv('churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
labelencoder_x2 = LabelEncoder()

x[:, 1]= labelencoder_x1.fit_transform(x[:, 1])
x[:, 2]= labelencoder_x2.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
x= onehotencoder.fit_transform(x).toarray()
x= x[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# **part2: make ANN 
import keras 
from keras.models import Sequential
from keras.layers import Dense

# **part3 -Making 


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)