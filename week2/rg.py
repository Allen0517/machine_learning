import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

fig = plt.figure()
data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv",index_col=0)

print(data.head())
print(data.tail())
print(data.shape)

sns.pairplot(data,x_vars=['TV','radio','newspaper'],y_vars='sales',size = 7, aspect = 0.7,kind ='reg')
fig.set_tight_layout(True)
plt.show()


features = ['TV','radio','newspaper']
X = data[features]
Y = data['sales']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state = 1)

linreg = LinearRegression()
linreg.fit(X_train,Y_train)

print(linreg.intercept_)
print(linreg.coef_)
zip(features,linreg.coef_)

Y_pred = linreg.predict(X_test)
print(metrics.mean_absolute_error(Y_test,Y_pred))
print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))


#remove newspaper
print("Newspaper removed")
features = ['TV','radio']
X = data[features]
Y = data['sales']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state = 1)

linreg = LinearRegression()
linreg.fit(X_train,Y_train)

print(linreg.intercept_)
print(linreg.coef_)
zip(features,linreg.coef_)

Y_pred = linreg.predict(X_test)
print(metrics.mean_absolute_error(Y_test,Y_pred))
print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))