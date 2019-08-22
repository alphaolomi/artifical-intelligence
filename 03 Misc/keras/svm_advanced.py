import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets.sample_generator import make_blobs

# we create 1 40 separable points
X, y = make_blobs(n_sample=40, centers=2, random_state= 20)

# fit the model
clf = scm.SVC(kernel='linear' , C=1)
clf.fit(X,y)

# Diaply the data i the graph
plt.scatter(X[:,0], X[:,1], c=y, s=30,cmap=plt.cm.Paired )
plt.show()
