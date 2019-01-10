#Maitreyee Mhasakar: mam171630 for Machine learning
#Problem Set 1 : Problem 1 Perceptron Learning

import csv
import numpy as np

#Stocastic Gradient Descent

# READING DATA
with open('perceptron.data') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    x_arr=[]    #training data set
    y_arr=[]    #training Output label 1 or -1
    for row in csv_reader:
        dimension=len(row[:-1]) #dimension of input data x
        x_arr.append(row[:-1])
        y_arr.append(row[-1])

        
x_arr=np.array(x_arr).astype('float')
y_arr=np.array(y_arr).astype('float')
W=[]
W0=np.array([0.0,0.0,0.0,0.0])
b=0
learn_rate=1



#PERCEPTRON LOSS
def perceptron_loss(x_arr,y_arr,learn_rate,W0,b):
    Lo=[]
    for it in range(len(x_arr)):
        Lo.append(max(0,-1*y_arr[it]*(np.dot(np.transpose(W0),x_arr[it])+b)))
        #Lo.append(max(0,-1*y_arr[it]*(np.dot(x_arr[it],W0)+b)))
    Summation_loss= sum(Lo)
    return Summation_loss


#Derivative of loss with respect to W

#Derivative of loss with respect to b

p_loss=perceptron_loss(x_arr,y_arr,learn_rate,W0,b)
print "Initial perceptron loss: ",p_loss

New_w=W0
New_b=b

for It in range(10):

    
    print "New perceptron loss",p_loss
    W_arr=[]
    b_arr=[]
    for it in range(len(x_arr)):
        constraint = -1*y_arr[it]*np.sign((np.dot(np.transpose(New_w),x_arr[it]))+ New_b) 
        
        if constraint >= 0 : # if point is misclassified, calculate gradient
            W_arr.append(y_arr[it]*x_arr[it])
            b_arr.append(y_arr[it])
    
            #W_arr1=sum(W_arr)
            #b_arr1=sum(b_arr)

            # Calculate new W and b
            New_w= New_w+learn_rate*y_arr[it]*x_arr[it]
            #print New_w
            New_b=New_b+learn_rate*y_arr[it]
            #print New_b
        p_loss=perceptron_loss(x_arr,y_arr,learn_rate,New_w,New_b)
    if 1<=It<=3:
        print "W for iteration ",It," is ",New_w
        print "W for iteration ",It,"is ",New_b
    

print "Perceptron loss is :",p_loss
print "Final W ",New_w
print "Final b",New_b

#Output for Standard Gradient Descent
'''

'''

    











