#Program to classify iris dataset using K NEAREST NEIGHBOR ALGORITHM 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  


df = pd.read_csv('iris.csv')

print(df.isnull().any())
print(df.dtypes)
print(df.describe())

all_inputs = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']].values

all_classes = df['Name'].values
(X_train, X_test, y_train, y_test) = train_test_split(all_inputs, all_classes, test_size=0.2, random_state=1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

#Classification
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)  

#Results
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

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
[[11  0  0]
 [ 0 12  1]
 [ 0  0  6]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      0.92      0.96        13
 Iris-virginica       0.86      1.00      0.92         6

    avg / total       0.97      0.97      0.97        30
'''