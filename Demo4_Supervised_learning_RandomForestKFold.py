#https://jupyter.org/try
#Demo4
#M. S. Rakha, Ph.D.
# Post-Doctoral - Queen's University 
# Supervised Learning - RandomForest Classification
# RandomForest Classification_Kfold
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

np.random.seed(5)
breastCancer = datasets.load_breast_cancer()

list(breastCancer.target_names)

#Only two features
X = breastCancer.data[:, 0:2]
y = breastCancer.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

X_train[:,0].size
X_train[:,0].size

varriableNames= breastCancer.feature_names
 

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold
kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None) 
randomForestModel = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

for train_index, test_index in kf.split(X):
    print("Train:", train_index, "Validation:",test_index)
    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = y[train_index], y[test_index]
    randomForestModel.fit(X_train, y_train)
    y_pred = randomForestModel.predict(X_test)
    print(classification_report(y_test, y_pred))

