Linear algorithms
===================

Linear algorithms are a common class of models that differ in their simplicity and speed of operation. They can be trained for a reasonable time on very large amounts of data, and at the same time they can work with any type of characteristics. Here, I will try to review and compare work of several linear algorithms.


Realization in scikit-learn
----------
Lets's start with [Perceptron](https://en.wikipedia.org/wiki/Perceptron). I will use the implementation of the library [scikit-learn](http://scikit-learn.org/stable/index.html). 
It is located in the package [sklearn.linear_model](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model), as a metric I will use the proportion of correct answers - [sklearn.metrics.accuracy_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).


```python
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

tr_data = pd.read_csv("train.csv", names=[1,2,3])
te_data = pd.read_csv("test.csv", names=[1,2,3])

tr_data = tr_data.as_matrix()

train_x = [[x[1], x[2]] for x in tr_data]
train_y = [x[0] for x in tr_data]


te_data = te_data.as_matrix()

test_x = [[x[1], x[2]] for x in te_data]
test_y = [x[0] for x in te_data]

clf_b = Perceptron(random_state=241)
clf_b.fit(train_x, train_y)
predicted_classes = clf_b.predict(test_x)
before_scale = accuracy_score(test_y, predicted) #0.654
```


  As in the case of metric methods, the quality of linear algorithms depends on some properties of the data, for example, the features should be normalized. Otherwise, the quality may fall.


This is the result of running the algorithm without scaling the features:


![before.png](https://github.com/AlievMagomed/ML-Perceptron-/blob/master/before.png?raw=true)


To scale features, it is convenient to use the class [sklearn.preprocessing.StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)

clf_a = Perceptron(random_state=241)
clf_a.fit(X_train_scaled, train_y)
predicted_classes = clf_a.predict(X_test_scaled)
after_scale = accuracy_score(test_y, predicted) #0.854
```


![after.png](https://github.com/AlievMagomed/ML-Perceptron-/blob/master/after.png?raw=true)

## Non-linear datasets

​	Perceptron cope with the task of binary classification pretty well, but it is clearly not suitable for linearly non-separable datasets. In that case, it is better to use [SVM](https://en.wikipedia.org/wiki/Support_vector_machine). In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the [kernel trick](https://en.wikipedia.org/wiki/Kernel_method), implicitly mapping their inputs into high-dimensional feature spaces.

​	Again, I will use scikit-learn. [SVM](http://scikit-learn.org/stable/modules/svm.html) classifier is located in [sklearn.svm](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm), many useful tools can be found in [sklearn.model_selection](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection): [train_test_split ](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) - split arrays or matrices into random train and test subsets, [StratifiedShuffleSplit ](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit) - provides train/test indices to split data in train/test sets and [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) - searches over specified parameter values for an estimator. This time I will use custom dataset, created with [make_circles](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html) of [sklearn.datasets](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets) class.

​	SVM has many parametrs we can interact with. It is very important to set up the classifier in a right way. Let's see how different settings can affect alorithm's work.

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.model_selection import StratifiedShuffleSplit


#creating non-linear dataset and and splitting it into training and testing parts
X, y = make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=241)
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#here I will consider only a small set of parametrs for visualization
C_range = [10, 100, 1000]
gamma_range = [0.001, 0.1, 10]

for C in C_range:
    for gamma in gamma_range:
        #setting up SVM with current settings
        clf = SVC(kernel='rbf', C=C, gamma=gamma)
        clf.fit(X_train, y_train)
        
        predicted = clf.predict(X_test)
        acc = accuracy_score(y_test, predicted)

```

![rbf_params](https://github.com/AlievMagomed/ML-Perceptron-/blob/master/RBF%20params.png?raw=true)

Of course, the search for the optimal combination of parameters can take a long time. In this case  GridSearchCV will help to simplify this process.

````python
#find best params using GridSearch with rbf kernel
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=241)
C_range = np.logspace(-5, 7, num=12)
gamma_range = np.logspace(-8, 3, num=11)
parametrs = dict(kernel=['rbf'], gamma=gamma_range, C=C_range)
grid = GridSearchCV(SVC(), param_grid=parametrs, cv=cv)
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %.2f"
      % (grid.best_params_, grid.best_score_))

#predict is now being called with best found params
predicted = grid.predict(X_test)
acc = accuracy_score(y_test, predicted)

print ("Accuracy of best-fitted estimator is %.2f" % acc)
````

```
The best parameters are {'kernel': 'rbf', 'C': 432.87612810830529, 'gamma': 0.039810717055349776} with a score of 0.88
Accuracy of best-fitted estimator is 0.88
```

Now let's compare the results of SVM and Perceptron to evaluate the advantages of this algorithm.

![compare](https://github.com/AlievMagomed/ML-Perceptron-/blob/master/rbf_perc_compare.png?raw=true)

