# Program using Support Vector Machine to classify Iris dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('iris.csv')

print(df.isnull().any())
print(df.dtypes)
print(df.describe())

all_inputs = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']].values

all_classes = df['Name'].values
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, test_size=0.3, random_state=0)
model=SVC()
model.fit(train_inputs, train_classes)
y_pred=model.predict(test_inputs)

cnf_matrix = metrics.confusion_matrix(test_classes, y_pred)
print(cnf_matrix)
print(classification_report(test_classes, y_pred))



'''
SepalLength    False
SepalWidth     False
PetalLength    False
PetalWidth     False
Name           False
dtype: bool
SepalLength    float64
SepalWidth     float64
PetalLength    float64
PetalWidth     float64
Name            object
dtype: object
       SepalLength  SepalWidth  PetalLength  PetalWidth
count   150.000000  150.000000   150.000000  150.000000
mean      5.843333    3.054000     3.758667    1.198667
std       0.828066    0.433594     1.764420    0.763161
min       4.300000    2.000000     1.000000    0.100000
25%       5.100000    2.800000     1.600000    0.300000
50%       5.800000    3.000000     4.350000    1.300000
75%       6.400000    3.300000     5.100000    1.800000
max       7.900000    4.400000     6.900000    2.500000
[[16  0  0]
 [ 0 17  1]
 [ 0  0 11]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        16
Iris-versicolor       1.00      0.94      0.97        18
 Iris-virginica       0.92      1.00      0.96        11

    avg / total       0.98      0.98      0.98        45




'''





