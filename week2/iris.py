from sklearn import neighbors, datasets
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris()
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)
print(type(iris.data))
print(iris.data.shape)

X = iris.data
Y = iris.target

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
print(knn)
knn.fit(X,Y)

X2 = [[3,5,4,2],[5,3,4,2]]
print(knn.predict(X2))

Y_pred =  knn.predict(X)
print(metrics.accuracy_score(Y,Y_pred))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.4)
print(X_train.shape)
print(X_test.shape)

knn.fit(X_train,Y_train)
Y_pred2 = knn.predict(X_test)
print(metrics.accuracy_score(Y_test,Y_pred2))

logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred3 = logreg.predict(X_test)
print(metrics.accuracy_score(Y_test,Y_pred3))

k_range = range(1,30)
scores = []


for k in k_range:
	knn = neighbors.KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,Y_train)
	Y2 = knn.predict(X_test)
	scores.append(metrics.accuracy_score(Y_test,Y2))

#print(scores)
plt.plot(k_range,scores)
plt.xlabel("Value of K in KNN")
plt.ylabel("Testing Accuracy")
plt.show()



