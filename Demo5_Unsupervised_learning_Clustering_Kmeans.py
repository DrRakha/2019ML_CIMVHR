#https://jupyter.org/try
#Demo5
#M. S. Rakha, Ph.D.
# Post-Doctoral - Queen's University 
# UnSupervised Learning - Clustering Kmeans
# Kmeans Clustering
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

np.random.seed(5)
breastCancer = datasets.load_breast_cancer()

list(breastCancer.target_names)

X = breastCancer.data[:, 0:2]

y = pd.DataFrame(breastCancer.target)
varriableNames= breastCancer.feature_names

#first ten records 
X[0:10,]

#Building you Kmeans model 
n_clusters = 2 # The number of clusters
init = 'random' # Centroids will be assigned in a random way
n_init = 10 # Number of iterations
clusteringKMeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init)
clusteringKMeans.fit(X)

##Plotting the model output
breastCancer_df =  pd.DataFrame(breastCancer.data)
breastCancer_df = breastCancer_df.iloc[:, 0:2]# first column of data frame (first_name)  
breastCancer_df.columns = ['meanRadius','meanTexture']  
y.columns = ["Targets"] 

color_theme = np.array(['red','darkgreen'])
plt.subplot(1,2,1)
plt.scatter(x=breastCancer_df.meanRadius, y=breastCancer_df.meanTexture,c=color_theme[breastCancer.target],s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=breastCancer_df.meanRadius, y=breastCancer_df.meanTexture,c=color_theme[clusteringKMeans.labels_],s=50)
plt.title('K-Means Clustering')
  
#Evaluate the model

print(classification_report(y,clusteringKMeans.labels_))
 

