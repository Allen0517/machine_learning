from sklearn import datasets, svm , neighbors
from sklearn.model_selection import cross_val_score, KFold, LeavePOut
import matplotlib.pyplot as plt
from sklearn import linear_model


output_file = "cross-validation-experiment-iris.png"

iris = datasets.load_iris()
X    = iris.data
y    = iris.target
clf  = svm.LinearSVC()
knn  = neighbors.KNeighborsClassifier(n_neighbors=10)
linreg = linear_model.LinearRegression()
n    = X.shape[0]

splits      = [2, 5, 10, 20, 30]
n_splits    = len(splits)
accuracies = []

# Run K-folds
print("SVM")
for k in splits:
    cv     = KFold(n_splits=k) if k > 0 else LeavePOut(p=abs(k))
    scores = cross_val_score(clf, X, y, cv=cv)
    accuracies.append(100 * scores.mean())
    print("K = %d, accuracy: %0.2f%%" % (k, accuracies[-1]))

print("KNN")
for k in splits:
    cv     = KFold(n_splits=k) if k > 0 else LeavePOut(p=abs(k))
    scores = cross_val_score(knn, X, y, cv=cv)
    accuracies.append(100 * scores.mean())
    print("K = %d, accuracy: %0.2f%%" % (k, accuracies[-1]))

