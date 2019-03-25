
import pandas as pd
from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('iris.csv')

print(df.isnull().any())
print(df.dtypes)
print(df.describe())

all_inputs = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']].values

all_classes = df['Name'].values
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, test_size=0.25, random_state=0)



logistic = LogisticRegression()
logistic.fit(train_inputs,train_classes)
y_pred=logistic.predict(test_inputs)
cnf_matrix = metrics.confusion_matrix(test_classes, y_pred)
print(cnf_matrix)
print(classification_report(test_classes, y_pred))
print(logistic.score(test_inputs,test_classes)*100)

''' 
OUTPUT

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
[[13  0  0]
 [ 0 11  5]
 [ 0  0  9]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        13
Iris-versicolor       1.00      0.69      0.81        16
 Iris-virginica       0.64      1.00      0.78         9

    avg / total       0.92      0.87      0.87        38

86.8421052631579

'''