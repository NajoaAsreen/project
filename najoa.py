# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:18:45 2019

@author: Najoa
"""

########################### Imorting Liabraries ###############################
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
############################# Data Analysis ###################################
data = pd.read_csv('najoa.csv')
print(data.shape)
print(data.keys())
print("\n\n")
#print('My dataset has {} data points with {} variables each.'.format(*data.shape))
y = data['Target']
x=data.drop(['Target'],1)
print(x.shape[0])
print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)
print (X_train.shape)
print (X_test.shape)

print (y_train.shape)
print (y_test.shape)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


########################## K-Nearest Neighbors ################################
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)

y_train_predKNN=clf.predict(X_train)
y_predKNN = clf.predict(X_test)
acc_KNN=metrics.accuracy_score(y_predKNN, y_test)
acc_KNN_train=metrics.accuracy_score(y_train, y_train_predKNN)

print('Accuracy : ')
print(acc_KNN)

print(x.shape)
print(y.shape)
plt.scatter(x['CGPA'],x['Leisure Time in Hour'])
plt.show()
plt.scatter(data['CGPA'], data['Leisure Time in Hour'], alpha=0.5, s=50*data['Future Plan'], c=data['Target'], cmap='viridis')
# s : scalar or array_like, shape (n, ), optional # The marker size in points**2. 
#


fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(x);
print(x['CGPA'])



#plt.xlabel(x['CGPA'])
#plt.ylabel(x['Leisure Time in Hour']);
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

#print(features)

#plt.scatter(features[0], features[1], alpha=0.5, s=100*features[3], c=iris.target, cmap='viridis')
# s : scalar or array_like, shape (n, ), optional # The marker size in points**2. 
#
#plt.xlabel(iris.feature_names[0])
#plt.ylabel(iris.feature_names[1]);
print('Actual')
print(y_test)
print('Predicted')
print(y_predKNN)
cm = confusion_matrix(y_test,y_predKNN)
print('test')
print(cm)