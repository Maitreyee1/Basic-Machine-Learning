#Maitreyee Mhasakar: mam171630 for Machine learning
#Problem Set 1 : Problem 1 Perceptron Learning

import csv
import numpy as np

#Standard Gradient Descent

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

for It in range(47):
    p_loss=perceptron_loss(x_arr,y_arr,learn_rate,New_w,New_b)
    print "New perceptron loss",p_loss
    W_arr=[]
    b_arr=[]
    for it in range(len(x_arr)):
        constraint = -1*y_arr[it]*np.sign((np.dot(np.transpose(New_w),x_arr[it]))+ New_b) 
        
        if constraint >= 0 : # if point is misclassified, calculate gradient
            W_arr.append(y_arr[it]*x_arr[it])
            b_arr.append(y_arr[it])
    
    W_arr1=sum(W_arr)
    b_arr1=sum(b_arr)

    # Calculate new W and b
    New_w= New_w+learn_rate*W_arr1
    New_b=New_b+learn_rate* b_arr1
    if 1<=It<=3:
        print "W for iteration ",It," is ",New_w
        print "W for iteration ",It,"is ",New_b
    

print "Perceptron loss is :",p_loss
print "Final W ",New_w
print "Final b",New_b

#Output for Standard Gradient Descent
'''
Initial perceptron loss:  0
New perceptron loss 0
New perceptron loss 184344.101515
W for iteration  1  is  [ 1307.29472974   432.74778799   -27.55191988 -1523.78895446]
W for iteration  1 is  -493.0
New perceptron loss 143254.416014
W for iteration  2  is  [ 1255.18981362   425.50402882    18.7965404  -1434.66754197]
W for iteration  2 is  -625.0
New perceptron loss 115551.319128
W for iteration  3  is  [ 1177.63614418   405.9535502     27.36183333 -1377.12882218]
W for iteration  3 is  -741.0
New perceptron loss 93557.8249433
New perceptron loss 73291.358954
New perceptron loss 54783.2640533
New perceptron loss 39643.9875412
New perceptron loss 28512.8204447
New perceptron loss 20867.8664096
New perceptron loss 15293.3961195
New perceptron loss 11306.9312693
New perceptron loss 8381.87289603
New perceptron loss 6195.09851221
New perceptron loss 4450.78885741
New perceptron loss 3160.03508255
New perceptron loss 2067.90746658
New perceptron loss 1252.06930939
New perceptron loss 744.510283321
New perceptron loss 480.062565799
New perceptron loss 232.072255671
New perceptron loss 293.800010586
New perceptron loss 260.475283329
New perceptron loss 56.0862900425
New perceptron loss 40.7143264893
New perceptron loss 30.7943192001
New perceptron loss 65.4716971516
New perceptron loss 5.53798303174
New perceptron loss 0.731609304907
New perceptron loss 13.3008446258
New perceptron loss 39.5317850292
New perceptron loss 6.76570967113
New perceptron loss 1.02685428042
New perceptron loss 5.12754238814
New perceptron loss 11.6319628279
New perceptron loss 72.4451379009
New perceptron loss 19.7450097779
New perceptron loss 15.2105813435
New perceptron loss 9.47172595276
New perceptron loss 7.00998561309
New perceptron loss 7.73161333686
New perceptron loss 1.99275794616
New perceptron loss 2.89149298298
New perceptron loss 5.068491756
New perceptron loss 2.63232824611
New perceptron loss 3.97914831961
New perceptron loss 0
Perceptron loss is : 0
Final W  [ 685.79932892  243.89947473    8.24199193 -797.62505314]
Final b -1485.0
'''

    











