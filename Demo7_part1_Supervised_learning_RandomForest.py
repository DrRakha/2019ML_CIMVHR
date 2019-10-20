#https://jupyter.org/try
#Demo7 - part1
#M. S. Rakha, Ph.D.
# Post-Doctoral - Queen's University 
#  
# Feature Selection#1 RandomForest
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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


#Univariate Selection
#Statistical tests can be used to select those features that have the strongest relationship with the output variable.
# example below uses the chi squared (chi^2) statistical test for non-negative features
np.random.seed(5)
breastCancer = datasets.load_breast_cancer()

list(breastCancer.target_names)

#Only two features
X = breastCancer.data
y = breastCancer.target



# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])



## Recursive Feature Elimination
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
from sklearn.feature_selection import RFE
np.random.seed(5)
breastCancer = datasets.load_breast_cancer()

list(breastCancer.target_names)

X = breastCancer.data
y = breastCancer.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

X_train[:,0].size
X_train[:,0].size

varriableNames= breastCancer.feature_names
  
#Feature extraction
#Recursive feature elimination (RFE) is a feature selection method that fits a model 
#and removes the weakest feature (or features) until the specified number of features is reached. 
#Features are ranked by the model’s coef_ or feature_importances_

randomForestModel = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rfe = RFE(randomForestModel, 3)
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)  
