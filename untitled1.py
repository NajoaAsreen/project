# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:40:26 2019

@author: Najoa
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

data = pd.read_csv('najoa.csv')

print(data.shape)
print(data.keys())
print("\n\n")

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

print('Actual')
print(y_test)
print('Predicted')
print(y_predKNN)
cm = confusion_matrix(y_test,y_predKNN)
print('test')
print(cm)

print(x.shape)
print(y.shape)
plt.scatter(x['CGPA'],x['Leisure Time in Hour'])
plt.xlabel('CGPA')
plt.ylabel('Leisure Time in Hour')
plt.show()
#plt.scatter(data['CGPA'], data['Leisure Time in Hour'], alpha=0.5, s=50*data['Future Plan'], c=data['Target'], cmap='viridis')
#plt.scatter(data['Target'], data['Leisure Time in Hour'], alpha=0.5, s=50*data['Future Plan'], c=data['CGPA'], cmap='viridis')


plt.scatter(x['Future Plan'],x['Leisure Time in Hour'])
plt.xlabel('Future Plan')
plt.ylabel('Leisure Time in Hour')
plt.show()



plt.scatter(data['Future Plan'], data['Leisure Time in Hour'], alpha=0.5, s=50*data['CGPA'], c=data['Target'], cmap='viridis')
plt.xlabel('Future Plan')
plt.ylabel('Leisure Time in Hour')


fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot of Leisure Time in Hour')

ax1.boxplot(x['Leisure Time in Hour'])

fig2, ax2 = plt.subplots()
ax2.set_title('Basic Plot of Future Plan')

ax2.boxplot(x['Future Plan'])



# Scatter the points, using size and color but no label
#plt.scatter(data['Future Plan'],data[' Leisure Time in Hour'], c=np.log10(data['CGPA']), cmap='viridis', s=data['Target'], linewidth=0, alpha=0.5)

#plt.axis(aspect='equal')
#plt.xlabel('Future Plan')
#plt.ylabel('Leisure Time in Hour')
#plt.colorbar(label='log$_{10}$data['CGPA'])')
#plt.clim(3, 7) #Set the color limits of the current image

#for area in [100, 300, 500]:
#    plt.scatter([], [], c='k', alpha=0.3, s=data['Target'], label=str(data['Target']) )

#plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')


#plt.title('California Cities: Area and Population');



#plt.scatter(data['Leisure Time in Hour'], data['Future Plan'], alpha=0.5, s=50*data['CGPA'], c=data['Target'], cmap='viridis')