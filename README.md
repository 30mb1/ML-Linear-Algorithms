Linear algorithms
===================

Linear algorithms are a common class of models that differ in their simplicity and speed of operation. They can be trained for a reasonable time on very large amounts of data, and at the same time they can work with any type of characteristics. Here, I will consider [perceptron](https://en.wikipedia.org/wiki/Perceptron) - one of the versions of linear models for binary classification.


Realization in scikit-learn
----------
I will use the perceptron implementation of the library [scikit-learn](http://scikit-learn.org/stable/index.html). 
It is located in the package [sklearn.linear_model](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model), as a metric I will use the proportion of correct answers - [sklearn.metrics.accuracy_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).


```
#!python

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


```
#!python
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
