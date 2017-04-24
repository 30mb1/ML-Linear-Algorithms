import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV


tr_data = pd.read_csv("train.csv", names=[1,2,3])
te_data = pd.read_csv("test.csv", names=[1,2,3])


tr_data = tr_data.as_matrix()

train_x = [[x[1], x[2]] for x in tr_data]
train_y = [x[0] for x in tr_data]


te_data = te_data.as_matrix()

test_x = [[x[1], x[2]] for x in te_data]
test_y = [x[0] for x in te_data]


plt.figure(figsize=(10, 6))
cm_bright = ListedColormap(['#0000FF', '#FF0000'])


clf_b = Perceptron(random_state=241)
clf_b.fit(train_x, train_y)

plot_x = [x[0] for x in test_x]
plot_y = [x[1] for x in test_x]

x_min, x_max = min(plot_x) - 1, max(plot_x) + 1
y_min, y_max = min(plot_y) - 1, max(plot_y) + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
					 np.arange(y_min, y_max, 2))
z = clf_b.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(plot_x, plot_y, c=test_y, cmap=cm_bright)
plt.title('Before scaling features \n accuracy = 66%')

#plt.show()

plt.figure(figsize=(10, 6))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)


clf_a = Perceptron(random_state=241)
clf_a.fit(X_train_scaled, train_y)


plot_x = [x[0] for x in X_test_scaled]
plot_y = [x[1] for x in X_test_scaled]

x_min, x_max = min(plot_x) - 1, max(plot_x) + 1
y_min, y_max = min(plot_y) - 1, max(plot_y) + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
					 np.arange(y_min, y_max, 0.02))

z = clf_a.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(plot_x, plot_y, c=test_y, cmap=cm_bright)

plt.title('After scaling features \n accuracy = 85%')


predicted = clf_b.predict(test_x)
before_scale = accuracy_score(test_y, predicted) #0.654

predicted = clf_a.predict(X_test_scaled)
after_scale = accuracy_score(test_y, predicted) #0.854

plt.show()


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.model_selection import StratifiedShuffleSplit

##plt.figure(figsize=(10, 6))


#creating non-linear dataset and and splitting it into training and testing parts
X, y = make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=241)
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



#find best params using GridSearch with rbf and polynomial kernels
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=241)
C_range = np.logspace(-5, 5, num=10)
gamma_range = np.logspace(-8, 3, num=11)
#degree_range = np.arange(1, 6)

parametrs = dict(kernel=['rbf'], gamma=gamma_range, C=C_range)
grid = GridSearchCV(SVC(), param_grid=parametrs, cv=cv)
grid.fit(X_train, y_train)


print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))



x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), 
	np.arange(y_min, y_max, 0.02))

#comparing results of Perceptron and SVM
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.xticks(())
plt.yticks(())

z = grid.predict(np.c_[xx.ravel(), yy.ravel()])
predicted = grid.predict(X_test)
z = z.reshape(xx.shape)
plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright)
plt.title("SVM with RBF kernel. Accuracy = %.2f" % (accuracy_score(y_test, predicted)))

plt.subplot(1, 2, 2)
plt.xticks(())
plt.yticks(())
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
predicted = clf.predict(X_test)
z = z.reshape(xx.shape)
plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright)
plt.title("Perceptron. Accuracy = %.2f" % (accuracy_score(y_test, predicted)))

plt.show()

#vizualizing SVM with different params
plt.figure(figsize=(12, 8))

C_range = [0.1, 1, 10]
gamma_range = [0.01, 0.1, 1]

k = 1
for C in C_range:
	for gamma in gamma_range:
		plt.subplot(len(C_range), len(gamma_range), k)
		plt.xticks(())
		plt.yticks(())
		k = k + 1
		clf = SVC(kernel='rbf', C=C, gamma=gamma)
		clf.fit(X_train, y_train)
		z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		z = z.reshape(xx.shape)
		plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
		plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright)
		predicted = clf.predict(X_test)
		acc = accuracy_score(y_test, predicted)
		plt.title("gamma=%.2f, C=%.2f acc=%.2f" % (gamma, C, acc), size='medium')


plt.show()
