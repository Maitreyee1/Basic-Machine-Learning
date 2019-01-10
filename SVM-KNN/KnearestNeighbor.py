'''
Maitreyee Mhasakar: net ID: mam171630
Problem Set 2: 
Problem 1: 
Following is the solution for 3. KNN

'''


#KNearestNeighbor for Parkinson Dataset
from __future__ import division
from collections import Counter
import csv
import random
import math
import operator
import numpy as np

def euclideanDistance(new_point, data_point, length):
    distance=0
    for x in range(length):
        distance += pow((float(new_point[x]) - float(data_point[x])), 2) #distance=((feature1_of_valid - feature1_of_training)^2+.......(featuren_valid - featuren_training)^2)
    return math.sqrt(distance) #squareroot(distance)

def findNeighbors(distances,ytrain,k):
    neighbour=[]
    for kiter in range(k):
        neighbour.append(distances[kiter])
    ylabel=[]
    for n in neighbour:
        ylabel.append(ytrain[n[1]])
    return ylabel


def predictlabel(ylabel):
    y_predict=max(ylabel,key=ylabel.count)
    c,number=Counter(ylabel).most_common(1)[0]
    return y_predict




def Accuracy(count,size):
    accuracy=(count/size)*100.0
    print "Accuracy ",accuracy
    
    
    




def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    validationSet=[]
    count=0
    count1=0
    with open('park_train.data', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        trainingSet = list(lines) #Training Data
    trainingSet=np.array(trainingSet) #Convert data into matrix


    with open('park_test.data', 'rb') as csvfile1:
        lines = csv.reader(csvfile1)
        testSet = list(lines)       #Test data

    with open('park_validation.data', 'rb') as csvfile1:
        lines = csv.reader(csvfile1)
        validationSet = list(lines)  #Validation data



# While testing use vlidationSet=np.array(testSet) and operate on the data
    validationSet=np.array(validationSet) #Convert data into matrix

        
    xvalid=validationSet[:,1:] #Features of validation data
    yvalid=validationSet[:,0]   #class Labels of validation data
    xtrain=trainingSet[:,1:]    #Features of training data
    ytrain=trainingSet[:,0]     #class labels of training data
    

    distances=[]

    size=len(xvalid)
    k=21
    print "k",k
    
    for i in range(size): # For number of data points in th given data
        arr_valid=[]
        arr_valid=xvalid[i] # 1 data row of xvalid
        arr_train=[]
        
        for j in range(len(xtrain)): #Comparing for all data points in training data
            arr_train=xtrain[j]
            dist=euclideanDistance(arr_valid,arr_train,len(arr_train)) # Compare distances of all datapoints in training data for given point in dataset
            distances.append((dist,j))
        distances=dict(distances)
        distances=sorted(distances.iteritems(), key=operator.itemgetter(0))

        ylabel=[]
        

        ylabel=findNeighbors(distances,ytrain,k)

        y_predict=predictlabel(ylabel)

        #Check correctness         
        if yvalid[i]==y_predict: # if label predicted by KNN is ame as in given dataset; increment counter
            
            count+=1
        
##        if yvalid[i]==ypredictbycounter:
##            count1+=1


    Accuracy(count,size)
        

            

            
            
main()
