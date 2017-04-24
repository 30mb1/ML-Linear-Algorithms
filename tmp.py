import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.model_selection import StratifiedShuffleSplit


scaler = StandardScaler()


X, y = make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=241)
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), 
	np.arange(y_min, y_max, 0.02))

clf = SVC(kernel='poly', C=100, gamma=10, degree=3)
clf.fit(X_train, y_train)

z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright)

predicted = clf.predict(X_test)
acc = accuracy_score(y_test, predicted)
plt.title("gamma=%.2f, degree=%.2f %.2f" % (gamma, degree, acc), size='medium')

plt.show()