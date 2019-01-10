'''
Maitreyee Mhasakar: net ID: mam171630
Problem Set 2: 
Problem 1: 
Following is the solution for 1. Primal SVM

'''


import numpy as np
import csv
from cvxopt import matrix,solvers

file=['park_train.data','park_validation.data','park_test.data']
data=list(csv.reader(open(file[0]),delimiter=','))
data=np.array(data).astype('float')
#x0=np.ones([len(data),1])
#data=np.append(x0,data,axis=1)

def SVM_slack(X,y,m,n,c):
    temp = matrix(np.hstack((2*np.eye(n),np.zeros(shape=(n,m+1)))))
    # ##print(a.size)
    # #
    P=matrix(np.vstack((temp,np.zeros(shape=(m+1,m+n+1)))))
    print (P.size)
    # print(P.size)
    
    q1=matrix(np.zeros(n))
    q2=matrix(c*np.ones(m))
    q3=matrix(np.zeros(shape=(1,1)))
    
    
    q= matrix(np.vstack((q1,q2,q3)))
    print (q.size)
    
    Xy=X*(-y[:,np.newaxis])
    Gtemp=matrix(Xy)
    print (Gtemp.size)
    si= matrix(-1*np.eye(m))
    b=matrix(-1*y)
    # #
    gx=matrix(np.zeros(shape=(m,n)))
    # gz=np.zeros(shape=(m,1))
    # #
    G1= matrix(np.vstack((Gtemp,gx)))
    # ##print(G1.size)
    # #
    G2=matrix(np.vstack((si,si)))
    # ##print(G2.size)
    # #
    G3=matrix(np.vstack((b,np.zeros(shape=(m,1)))))
    # ##print(G3.size)
    # #
    G=matrix(np.hstack((G1,G2,G3)))
    
    print (G.size)
    # #
    h1=matrix(-1*np.ones(m))
    h2=matrix(np.zeros(m))
    h=matrix(np.vstack((h1,h2)))
    print(h.size)
    
    sol=solvers.qp(P,q,G,h)
    solution=[]
    solution=sol['x']
    W=[]
    for i in range(n):
        W.append(solution[i])
    #print ("W",W)
    
    print("W shape",len(W))
    
    W1=np.array(W)
    print (W1.shape)
    W1=np.transpose(W1)
    print (W1.shape)
    
    print(X.shape)
    b=solution[-1]
    #print (X)
    
    A=np.dot(W1,np.transpose(X))
    print("A shape",A.shape)
    
    Atemp=[]
    for i in range(m):
        Atemp.append((A[i]+b))
    print (Atemp)
    predict_label=[]
    for i in range(m):
        if Atemp[i]>0:
            predict_label.append(1)
        if Atemp[i]<0:
            predict_label.append(-1)
    #print(predict_label)
    
    #print(len(predict_label))
    correct=0
    for i in range(m):
        if predict_label[i]==y[i]:
            correct+=1
    #print("Correct Labels",correct)
    
    Accuracy=(correct/m)*100
    print("Accuracy",Accuracy)
    return Accuracy
    









X = data[:,1:]	
y = data[:,0]
#print(y)	

for i in range(len(y)):
    if y[i]==0:
        y[i]=-1
    else:
        y[i]=1
#print(y)

c=100
m,n = X.shape
print(m,n)

# =============================================================================
# #
# ##SVM with slack
# #
 
C=[1,10,100,1000,10000,100000,1000000,10000000,100000000]
Accuracy_arr=[]

for i in range(len(C)):
    Accuracy=SVM_slack(X,y,m,n,C[i])
    Accuracy_arr.append(Accuracy)
print(Accuracy_arr)































# 
