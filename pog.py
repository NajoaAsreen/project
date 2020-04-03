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
import numpy as np
import pandas as pd
############################# Data Analysis ###################################
data = pd.read_csv('test.csv')
print(data.shape)
print(data.keys())
print("\n\n")
#print('My dataset has {} data points with {} variables each.'.format(*data.shape))
y = data['goals']
x=data.drop(['goals'],1)
#print(x.shape[0])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)
print (X_train.shape)
print (X_test.shape)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

########################### Random Forest Classifier ##########################
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)

y_train_predRFC = clf.predict(X_train)
y_predRFC = clf.predict(X_test)
acc_RFC=metrics.accuracy_score(y_predRFC, y_test)
acc_RFC_train=metrics.accuracy_score(y_train, y_train_predRFC)
mse_RFC=metrics.mean_squared_error(y_test, y_predRFC)

########################## Stochastic Gradient Decent #########################
clf = SGDClassifier()
clf.fit(X_train, y_train)
    
y_train_predSGD = clf.predict(X_train)
y_predSGD = clf.predict(X_test)
acc_SGD=metrics.accuracy_score(y_predSGD, y_test)
acc_SGD_train=metrics.accuracy_score(y_train, y_train_predSGD)
mse_SGD=metrics.mean_squared_error(y_test, y_predSGD)

######################### Support Vector Machine ##############################
clf = SVC()
clf.fit(X_train, y_train)

y_train_predSVM=clf.predict(X_train)
y_predSVM = clf.predict(X_test)
acc_SVM=metrics.accuracy_score(y_predSVM, y_test)
acc_SVM_train=metrics.accuracy_score(y_train, y_train_predSVM)
mse_SVM=metrics.mean_squared_error(y_test, y_predSVM)

########################## K-Nearest Neighbors ################################
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)

y_train_predKNN=clf.predict(X_train)
y_predKNN = clf.predict(X_test)
acc_KNN=metrics.accuracy_score(y_predKNN, y_test)
acc_KNN_train=metrics.accuracy_score(y_train, y_train_predKNN)
mse_KNN=metrics.mean_squared_error(y_test, y_predKNN)

########################## Decision Tree ######################################
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_train_predDT=clf.predict(X_train)
y_predDT = clf.predict(X_test)
acc_DT=metrics.accuracy_score(y_predDT, y_test)
acc_DT_train=metrics.accuracy_score(y_train, y_train_predDT)
mse_DT=metrics.mean_squared_error(y_test, y_predDT)

########################## Decision Tree Graph Plot ###########################
'''
plt.scatter(y_test, y_predDT)
plt.xlabel("Goals: $Y_i$")
plt.ylabel("Predicted goals: $\hat{Y}_i$")
plt.title("Goals vs Predicted goals: $Y_i$ vs $\hat{Y}_i$")
'''
###############################################################################

######################### Multi-layer Perceptron ##############################
clf = MLPClassifier(solver='adam',hidden_layer_sizes=350,alpha=1e-04)
clf.fit(X_train, y_train)

y_train_predMLP=clf.predict(X_train)
y_predMLP = clf.predict(X_test)
acc_MLP=metrics.accuracy_score(y_predMLP, y_test)
acc_MLP_train=metrics.accuracy_score(y_train, y_train_predMLP)
mse_MLP=metrics.mean_squared_error(y_test, y_predMLP)

######################### AdaBoostClassifier ##################################
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_train_predABC=clf.predict(X_train)
y_predABC = clf.predict(X_test)
acc_ABC=metrics.accuracy_score(y_predMLP, y_test)
acc_ABC_train=metrics.accuracy_score(y_train, y_train_predMLP)
mse_ABC=metrics.mean_squared_error(y_test, y_predMLP)

############################ PRINT ############################################
####################### Accuracy Table ########################################
print("\nTable For Training Accuracy : ")
models = pd.DataFrame({
    'Model': ['Random Forest CLassifier','Stochastic Gradient Decent',
              'Support Vector Machine','K-Nearest Neighbors',
              'Decision Tree','Multi-layer Perceptron',
              'AdaBoostClassifier'],
    'Score': [acc_RFC_train, acc_SGD_train,acc_SVM_train,acc_KNN_train,
              acc_DT_train,acc_MLP_train,acc_ABC_train]})
models.sort_values(['Score'], inplace=True,ascending=False)
print("\n",models,"\n\n")

########

print("\nTable For Testing Accuracy : ")
models = pd.DataFrame({
    'Model': ['Random Forest CLassifier','Stochastic Gradient Decent',
              'Support Vector Machine','K-Nearest Neighbors',
              'Decision Tree','Multi-layer Perceptron',
              'AdaBoostClassifier'],
    'Score': [acc_RFC, acc_SGD,acc_SVM,acc_KNN,acc_DT,acc_MLP,acc_ABC]})
models.sort_values(['Score'], inplace=True,ascending=False)
print("\n",models,"\n\n")

###############################################################################
############################# Data Analysis ###################################
data = pd.read_csv('test.csv')
#print(data.shape)
#print(data.keys())
#print("\n\n")
y = data['goals'].values
data.drop('goals',axis=1, inplace=True)
X = data.values
##############################Classifier Object################################
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier(random_state=0)
svc = svm.SVC(kernel='linear',C=0.4)
knn = KNeighborsClassifier(n_neighbors=5)
abc = AdaBoostClassifier(n_estimators=100)
sgd = SGDClassifier()
mlp = MLPClassifier(solver='adam',hidden_layer_sizes=350,alpha=1e-04)
##############################List Declareation################################
accuracy_rfc = []
accuracy_dtc = []
accuracy_svc = []
accuracy_knn = []
accuracy_abc = []
accuracy_sgd = []
accuracy_mlp = []

precision_rfc = []
precision_dtc = []
precision_svc = []
precision_knn = []
precision_abc = []
precision_sgd = []
precision_mlp = []

recall_rfc = []
recall_dtc = []
recall_svc = []
recall_knn = []
recall_abc = []
recall_sgd = []
recall_mlp = []


F1_rfc = []
F1_dtc = []
F1_svc = []
F1_knn = []
F1_abc = []
F1_sgd = []
F1_mlp = []

################################K Fold Cross Validation########################
kf = KFold(n_splits=5,random_state = 0,shuffle = True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = preprocessing.StandardScaler().fit(X_train.astype(float))     
    
    X_train = scaler.transform(X_train.astype(float))
    X_test = scaler.transform(X_test.astype(float))
    
    rfc.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    abc.fit(X_train, y_train)
    sgd.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    
    y_pred_rfc = rfc.predict(X_test)
    y_pred_dtc = dtc.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    y_pred_abc = abc.predict(X_test)
    y_pred_sgd = sgd.predict(X_test)
    y_pred_mlp = mlp.predict(X_test)
    
    accuracy_rfc.append( metrics.accuracy_score(y_test, y_pred_rfc))
    accuracy_dtc.append( metrics.accuracy_score(y_test, y_pred_dtc))
    accuracy_svc.append( metrics.accuracy_score(y_test, y_pred_svc))
    accuracy_knn.append( metrics.accuracy_score(y_test, y_pred_knn))
    accuracy_abc.append( metrics.accuracy_score(y_test, y_pred_abc))
    accuracy_sgd.append( metrics.accuracy_score(y_test, y_pred_sgd))
    accuracy_mlp.append( metrics.accuracy_score(y_test, y_pred_mlp))
    
    precision_rfc.append(metrics.precision_score(y_test, y_pred_rfc,average='macro'))
    precision_dtc.append(metrics.precision_score(y_test, y_pred_dtc,average='macro'))
    precision_svc.append(metrics.precision_score(y_test, y_pred_svc,average='macro'))
    precision_knn.append(metrics.precision_score(y_test, y_pred_knn,average='macro'))
    precision_abc.append(metrics.precision_score(y_test, y_pred_abc,average='macro'))
    precision_sgd.append(metrics.precision_score(y_test, y_pred_sgd,average='macro'))
    precision_mlp.append(metrics.precision_score(y_test, y_pred_mlp,average='macro'))
    
    recall_rfc.append(metrics.recall_score(y_test, y_pred_rfc,average='macro'))
    recall_dtc.append(metrics.recall_score(y_test, y_pred_dtc,average='macro'))
    recall_svc.append(metrics.recall_score(y_test, y_pred_svc,average='macro'))
    recall_knn.append(metrics.recall_score(y_test, y_pred_knn,average='macro'))
    recall_abc.append(metrics.recall_score(y_test, y_pred_abc,average='macro'))
    recall_sgd.append(metrics.recall_score(y_test, y_pred_sgd,average='macro'))
    recall_mlp.append(metrics.recall_score(y_test, y_pred_mlp,average='macro'))
    
    
    F1_rfc.append(metrics.f1_score(y_test, y_pred_rfc,average='macro'))
    F1_dtc.append(metrics.f1_score(y_test, y_pred_dtc,average='macro'))
    F1_svc.append(metrics.f1_score(y_test, y_pred_svc,average='macro'))
    F1_knn.append(metrics.f1_score(y_test, y_pred_knn,average='macro'))
    F1_abc.append(metrics.f1_score(y_test, y_pred_abc,average='macro'))
    F1_sgd.append(metrics.f1_score(y_test, y_pred_sgd,average='macro'))
    F1_mlp.append(metrics.f1_score(y_test, y_pred_mlp,average='macro'))


####################### Accuracy Table after K-FOLD########################################

print("\nTable For Accuracy After K-FOLD CrossValidation : ")
models = pd.DataFrame({
    'Model': ['Random Forest CLassifier','Decision Tree',
              'Support Vector Machine','K-Nearest Neighbors','AdaBoostClassifier',
              'Stochastic Gradient Decent','Multi-layer Perceptron'],
    'Score': [np.mean(accuracy_rfc),np.mean(accuracy_dtc),np.mean(accuracy_svc),
              np.mean(accuracy_knn),np.mean(accuracy_abc),np.mean(accuracy_sgd),
              np.mean(accuracy_mlp)]})
models.sort_values(['Score'], inplace=True,ascending=False)
print("\n",models,"\n\n")

###########################Performance metric##################################

print ("FOR Random Forest Classifier:")

print ("    Avg Accuracy : ",np.mean(accuracy_rfc))
print ("    Avg Precision : ",np.mean(precision_rfc))
print ("    Avg Recall : ",np.mean(recall_rfc))
print ("    Avg F1 : ",np.mean(F1_rfc))
print ("         ")
print ("         ")


print ("FOR Decision Tree Classifier:")

print ("    Avg Accuracy : ",np.mean(accuracy_dtc))
print ("    Avg Precision : ",np.mean(precision_dtc))
print ("    Avg Recall : ",np.mean(recall_dtc))
print ("    Avg F1 : ",np.mean(F1_dtc))
print ("         ")
print ("         ")


print ("FOR Support Vector Classification:")

print ("    Avg Accuracy : ",np.mean(accuracy_svc))
print ("    Avg Precision : ",np.mean(precision_svc))
print ("    Avg Recall : ",np.mean(recall_svc))
print ("    Avg F1 : ",np.mean(F1_svc))
print ("         ")
print ("         ")


print ("FOR KNeighbors Classifier:")

print ("    Avg Accuracy : ",np.mean(accuracy_knn))
print ("    Avg Precision : ",np.mean(precision_knn))
print ("    Avg Recall : ",np.mean(recall_knn))
print ("    Avg F1 : ",np.mean(F1_knn))
print ("         ")
print ("         ")

print ("FOR AdaBoost Classifier:")

print ("    Avg Accuracy : ",np.mean(accuracy_abc))
print ("    Avg Precision : ",np.mean(precision_abc))
print ("    Avg Recall : ",np.mean(recall_abc))
print ("    Avg F1 : ",np.mean(F1_abc))
print ("         ")
print ("         ")

print ("FOR Stochastic Gradient Decent Classifier:")

print ("    Avg Accuracy : ",np.mean(accuracy_sgd))
print ("    Avg Precision : ",np.mean(precision_sgd))
print ("    Avg Recall : ",np.mean(recall_sgd))
print ("    Avg F1 : ",np.mean(F1_sgd))
print ("         ")
print ("         ")

print ("FOR Multi-layer Perceptron:")

print ("    Avg Accuracy : ",np.mean(accuracy_mlp))
print ("    Avg Precision : ",np.mean(precision_mlp))
print ("    Avg Recall : ",np.mean(recall_mlp))
print ("    Avg F1 : ",np.mean(F1_mlp))
print ("         ")
print ("         ")


###############################################################################

import seaborn as sns
corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

##############
path='test.csv'
mpg_data = pd.read_csv(path, delim_whitespace=True, header=None,
            names = ['position', 'finishing', 'recent_scoring_from','opp_team_rating',
            'h_or_a','goals'],
            na_values='?')
mpg_data.drop(['goals'], axis=1).corr(method='spearman')

###############################################################################
