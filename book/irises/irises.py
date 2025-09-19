from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0) # try to increase random_state to see if the results are different

import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='0', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.savefig('iris_scatter_matrix.png')

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))