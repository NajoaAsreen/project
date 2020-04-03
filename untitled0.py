# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:52:28 2019

@author: Najoa
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:14:30 2019

@author: Najoa
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:40:26 2019

@author: Najoa
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
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

k_range = range(1,26)
scores_list=[]
scores={}
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)

    y_train_predKNN=knn.predict(X_train)
    y_predKNN = knn.predict(X_test)
  scores[k]=metrics.accuracy_score(y_test,y_predKNN)
   scores_list.append(metrics.accuracy_score(y_test,y_predKNN))
    
plt.plot(k_range,scores_list)
plt.xlabel('values of K for knn')
plt.ylabel('testing accuracy')
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




a1 = data['CGPA']
b1 =data['Leisure Time in Hour']
c1= data['Future Plan']
df = pd.DataFrame({'x': a1, 'y': b1, 'z':c1})
df = pd.DataFrame({ 'CGPA':a1, 'Leisure Time in Hour':b1, 'Future Plan':c1})
df.plot.box(grid='True')
ax = df.plot.kde()



