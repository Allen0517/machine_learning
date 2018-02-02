from sklearn import svm, datasets
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split




iris = load_iris()
print(iris.data)


X = iris.data
Y = iris.target

svm = svm.SVC()
print(svm)
svm.fit(X,Y)

X2 = [[3,5,4,2],[5,3,4,2]]
print(svm.predict(X2))



X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.4)
print(X_train.shape)
print(X_test.shape)

svm.fit(X_train,Y_train)
Y_pred2 = svm.predict(X_test)
print(metrics.accuracy_score(Y_test,Y_pred2))


