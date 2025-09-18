from sklearn.datasets import load_iris
iris_dataset = load_iris()
print(iris_dataset.keys())
# print(iris_dataset['DESCR'])
print(iris_dataset['feature_names'])
# print(iris_dataset['target'])
# print(iris_dataset['target_names'])
# print(iris_dataset['data'])
# print(iris_dataset['data'].shape)


# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)

# print(accuracy_score(y_test, clf.predict(X_test)))
