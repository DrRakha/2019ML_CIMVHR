#https://jupyter.org/try
#Demo7 - part2
#M. S. Rakha, Ph.D.
# Post-Doctoral - Queen's University 
# Supervised Learning - Random Forest
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

np.random.seed(5)
breastCancer = datasets.load_breast_cancer()

list(breastCancer.target_names)

#Only two features
X = breastCancer.data[:, 0:10]
y = breastCancer.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

X_train[:,0].size
X_train[:,0].size

varriableNames= breastCancer.feature_names
 

 

randomForestModel = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

randomForestModel.fit(X_train, y_train);

y_pred = randomForestModel.predict(X_test)


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))



importances = randomForestModel.feature_importances_
std = np.std([tree.feature_importances_ for tree in randomForestModel.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

print(varriableNames)
