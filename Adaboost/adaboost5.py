'''
Maitreyee Mhasakar: net ID: mam171630
Problem Set 3: 
Problem 2: 
Boosting Decision Trees on Heart Dataset

'''

from __future__ import division
import csv
import numpy as np
import math

# READING DATA
with open('heart_train.data') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    dataset=[]
    x_arr=[]    #training data set
    y_arr=[]    #training Output label 1 or -1
    for row in csv_reader:
        dataset.append(row)
        dimension=len(row[1:]) #dimension of input data x
        
        x_arr.append(row[1:])
        y_arr.append(row[0])

#print len(x_arr[0)



def treegenerator(f1,f2,f3,label):
    tree1={f1:{1:label[0],0:{f2:{1:label[1],0:{f3:{1:label[2],0:label[3]}}}}}}
    tree2={f1:{1:label[0],0:{f2:{0:label[1],1:{f3:{1:label[2],0:label[3]}}}}}}
    tree3={f1:{0:label[0],1:{f2:{1:label[1],0:{f3:{1:label[2],0:label[3]}}}}}}
    tree4={f1:{0:label[0],1:{f2:{0:label[1],1:{f3:{1:label[2],0:label[3]}}}}}}
    tree5={f1:{0:{f2:{1:label[0],0:label[1]}},0:{f3:{1:label[2],0:label[3]}}}}
    return tree1,tree2,tree3,tree4,tree5
    


def comparison(temp,tempo,f1,f2,f3):
    #print "temp",temp[f1]
    if isinstance(tempo[temp[f1]],dict):
        #print "Trueeee"
        tempo1=tempo[temp[f1]]
        #print tempo1
        #print temp[f2]
        var=temp[f2]
        #print tempo1[f2]
        tempo2=tempo1[f2]
        if isinstance(tempo2[temp[f2]],dict):
            #print "Trueeee f2"
            tempo3=tempo2[temp[f2]]
            tempo4=tempo3[f3]
            return tempo4[temp[f3]]
                        
        else:
            #print "False F2"
            return tempo2[temp[f2]]

        
    else:
        #print "Falseee its ",tempo[temp[f1]]
        return tempo[temp[f1]]
 
    


# Error Calculation

def error_Calculation(tree,f1,f2,f3,weight_arr):
    error=0
    correct=0
    label_arr=[]
    for i in range(len(x_arr)):
        temp=x_arr[i]
        tempo=tree1[f1]
        label=comparison(temp,tempo,f1,f2,f3)
        label_arr.append(label)
        if y_arr[i]==label:
            correct+=1
            #print "Correct"
        else:
            error+=weight_arr[i]
            

    return error,label_arr
            
# Calculate Alpha                   
def computealpha(final_error):
    tempalpha=(1-final_error)/final_error
    logalpha=math.log(tempalpha)
    alpha=(1/2)*logalpha
    return alpha
    
               
               
#weight updation
def update_weights(final_error,alpha,weight_array,label):
    label=np.asarray(label)
    denominator= 2*math.sqrt(final_error*(1-final_error))
    updated_w_arr=[]

    
    for i in range(len(x_arr)):
        temp=-1*y_arr[i]*label[i]*alpha
        tempw=weight_array[i]*math.exp(temp)
        updated_weight= tempw/denominator
        updated_w_arr.append(updated_weight)

    return updated_w_arr





#Assign weights

x_arr=np.asarray(x_arr).astype('int')
y_arr=np.asarray(y_arr).astype('int')


weight= 1/len(x_arr)
#print x_arr

arr=[]
final_error=1000
count=0
weight_array=[]
for i in range(len(x_arr)):
    weight_array.append(weight)





#Hypothesis selection
for f1 in range(len(x_arr[0])):
    for f2 in range(len(x_arr[0])):
        for f3 in range(len(x_arr[0])):
            #print f1,",",f2,",",f3
            
            arr.append((f1,f2,f3))
            count+=1
            #print "c",count
            for li1 in range(0,2):
               for li2 in range(0,2):
                   for li3 in range(0,2):
                       for li4 in range(0,2):
                           label=[li1,li2,li3,li4]
                           tree1,tree2,tree3,tree4,tree5=treegenerator(f1,f2,f3,label)
                           tree=[tree1,tree2,tree3,tree4,tree5]
                           #label1=[]
                           #label2=[]
                           #label3=[]
                           #label4=[]
                           #label5=[]
                           #fin_label=[]
                           
                           error1,label1=error_Calculation(tree1,f1,f2,f3,weight_array)
                           error2,label2=error_Calculation(tree2,f1,f2,f3,weight_array)
                           error3,label3=error_Calculation(tree3,f1,f2,f3,weight_array)
                           error4,label4=error_Calculation(tree4,f1,f2,f3,weight_array)
                           error5,label5=error_Calculation(tree5,f1,f2,f3,weight_array)
                           error_arr=[error1,error2,error3,error4,error5]
                           label_arr=[label1,label2,label3,label4,label5]
                           error_val=min(error_arr)
                           error_index=error_arr.index(error_val)

                           if error_val<final_error:
                               final_error=error_val
                               best_tree=tree[error_index]
                               fin_label=label_arr[error_index]
##print best_tree
##print len(label1)
##print len(label2)
##print len(label3)
##print len(label4)
##print len(label5)
##print "error",error_arr
##print "label", len(fin_label)

alpha=computealpha(final_error)
print "Error",final_error
print "Alpha",alpha

weight_array=update_weights(final_error,alpha,weight_array,fin_label)
print weight_array

for f1 in range(len(x_arr[0])):
    for f2 in range(len(x_arr[0])):
        for f3 in range(len(x_arr[0])):
            #print f1,",",f2,",",f3
            
            arr.append((f1,f2,f3))
            count+=1
            #print "c",count
            for li1 in range(0,2):
               for li2 in range(0,2):
                   for li3 in range(0,2):
                       for li4 in range(0,2):
                           label=[li1,li2,li3,li4]
                           tree1,tree2,tree3,tree4,tree5=treegenerator(f1,f2,f3,label)
                           tree=[tree1,tree2,tree3,tree4,tree5]
                           #label1=[]
                           #label2=[]
                           #label3=[]
                           #label4=[]
                           #label5=[]
                           #fin_label=[]
                           
                           error1,label1=error_Calculation(tree1,f1,f2,f3,weight_array)
                           error2,label2=error_Calculation(tree2,f1,f2,f3,weight_array)
                           error3,label3=error_Calculation(tree3,f1,f2,f3,weight_array)
                           error4,label4=error_Calculation(tree4,f1,f2,f3,weight_array)
                           error5,label5=error_Calculation(tree5,f1,f2,f3,weight_array)
                           error_arr=[error1,error2,error3,error4,error5]
                           label_arr=[label1,label2,label3,label4,label5]
                           error_val=min(error_arr)
                           error_index=error_arr.index(error_val)

                           if error_val<final_error:
                               final_error=error_val
                               best_tree=tree[error_index]
                               fin_label=label_arr[error_index]
##print best_tree
##print len(label1)
##print len(label2)
##print len(label3)
##print len(label4)
##print len(label5)
##print "error",error_arr
##print "label", len(fin_label)

alpha=computealpha(final_error)
print "Error",final_error
print "Alpha",alpha

weight_array=update_weights(final_error,alpha,weight_array,fin_label)
print weight_array

for f1 in range(len(x_arr[0])):
    for f2 in range(len(x_arr[0])):
        for f3 in range(len(x_arr[0])):
            #print f1,",",f2,",",f3
            
            arr.append((f1,f2,f3))
            count+=1
            #print "c",count
            for li1 in range(0,2):
               for li2 in range(0,2):
                   for li3 in range(0,2):
                       for li4 in range(0,2):
                           label=[li1,li2,li3,li4]
                           tree1,tree2,tree3,tree4,tree5=treegenerator(f1,f2,f3,label)
                           tree=[tree1,tree2,tree3,tree4,tree5]
                           #label1=[]
                           #label2=[]
                           #label3=[]
                           #label4=[]
                           #label5=[]
                           #fin_label=[]
                           
                           error1,label1=error_Calculation(tree1,f1,f2,f3,weight_array)
                           error2,label2=error_Calculation(tree2,f1,f2,f3,weight_array)
                           error3,label3=error_Calculation(tree3,f1,f2,f3,weight_array)
                           error4,label4=error_Calculation(tree4,f1,f2,f3,weight_array)
                           error5,label5=error_Calculation(tree5,f1,f2,f3,weight_array)
                           error_arr=[error1,error2,error3,error4,error5]
                           label_arr=[label1,label2,label3,label4,label5]
                           error_val=min(error_arr)
                           error_index=error_arr.index(error_val)

                           if error_val<final_error:
                               final_error=error_val
                               best_tree=tree[error_index]
                               fin_label=label_arr[error_index]
##print best_tree
##print len(label1)
##print len(label2)
##print len(label3)
##print len(label4)
##print len(label5)
##print "error",error_arr
##print "label", len(fin_label)

alpha=computealpha(final_error)
print "Error",final_error
print "Alpha",alpha

weight_array=update_weights(final_error,alpha,weight_array,fin_label)
print weight_array

for f1 in range(len(x_arr[0])):
    for f2 in range(len(x_arr[0])):
        for f3 in range(len(x_arr[0])):
            #print f1,",",f2,",",f3
            
            arr.append((f1,f2,f3))
            count+=1
            #print "c",count
            for li1 in range(0,2):
               for li2 in range(0,2):
                   for li3 in range(0,2):
                       for li4 in range(0,2):
                           label=[li1,li2,li3,li4]
                           tree1,tree2,tree3,tree4,tree5=treegenerator(f1,f2,f3,label)
                           tree=[tree1,tree2,tree3,tree4,tree5]
                           #label1=[]
                           #label2=[]
                           #label3=[]
                           #label4=[]
                           #label5=[]
                           #fin_label=[]
                           
                           error1,label1=error_Calculation(tree1,f1,f2,f3,weight_array)
                           error2,label2=error_Calculation(tree2,f1,f2,f3,weight_array)
                           error3,label3=error_Calculation(tree3,f1,f2,f3,weight_array)
                           error4,label4=error_Calculation(tree4,f1,f2,f3,weight_array)
                           error5,label5=error_Calculation(tree5,f1,f2,f3,weight_array)
                           error_arr=[error1,error2,error3,error4,error5]
                           label_arr=[label1,label2,label3,label4,label5]
                           error_val=min(error_arr)
                           error_index=error_arr.index(error_val)

                           if error_val<final_error:
                               final_error=error_val
                               best_tree=tree[error_index]
                               fin_label=label_arr[error_index]
##print best_tree
##print len(label1)
##print len(label2)
##print len(label3)
##print len(label4)
##print len(label5)
##print "error",error_arr
##print "label", len(fin_label)

alpha=computealpha(final_error)
print "Error",final_error
print "Alpha",alpha

weight_array=update_weights(final_error,alpha,weight_array,fin_label)
print weight_array

for f1 in range(len(x_arr[0])):
    for f2 in range(len(x_arr[0])):
        for f3 in range(len(x_arr[0])):
            #print f1,",",f2,",",f3
            
            arr.append((f1,f2,f3))
            count+=1
            #print "c",count
            for li1 in range(0,2):
               for li2 in range(0,2):
                   for li3 in range(0,2):
                       for li4 in range(0,2):
                           label=[li1,li2,li3,li4]
                           tree1,tree2,tree3,tree4,tree5=treegenerator(f1,f2,f3,label)
                           tree=[tree1,tree2,tree3,tree4,tree5]
                           #label1=[]
                           #label2=[]
                           #label3=[]
                           #label4=[]
                           #label5=[]
                           #fin_label=[]
                           
                           error1,label1=error_Calculation(tree1,f1,f2,f3,weight_array)
                           error2,label2=error_Calculation(tree2,f1,f2,f3,weight_array)
                           error3,label3=error_Calculation(tree3,f1,f2,f3,weight_array)
                           error4,label4=error_Calculation(tree4,f1,f2,f3,weight_array)
                           error5,label5=error_Calculation(tree5,f1,f2,f3,weight_array)
                           error_arr=[error1,error2,error3,error4,error5]
                           label_arr=[label1,label2,label3,label4,label5]
                           error_val=min(error_arr)
                           error_index=error_arr.index(error_val)

                           if error_val<final_error:
                               final_error=error_val
                               best_tree=tree[error_index]
                               fin_label=label_arr[error_index]
#print best_tree
#print len(label1)
#print len(label2)
#print len(label3)
#print len(label4)
#print len(label5)
#print "error",error_arr
#print "label", len(fin_label)

alpha=computealpha(final_error)
print "Error",final_error
print "Alpha",alpha
weight_array=update_weights(final_error,alpha,weight_array,fin_label)
print weight_array

                           


##f1=2
##f2=4
##f3=8
##f4=8
##label=[1,0,1,0]
##tree1,tree2,tree3,tree4,tree5=treegenerator(f1,f2,f3,label)
##error1=error_Calculation(tree1,f1,f2,f3)
##error2=error_Calculation(tree2,f1,f2,f3)
##error3=error_Calculation(tree3,f1,f2,f3)
##error4=error_Calculation(tree4,f1,f2,f3)
##error5=error_Calculation(tree5,f1,f2,f3)
##error_arr=[error1,error2,error3,error4,error5]
##print error_arr
##error_val=min(error_arr)
##error_index=error_arr.index(error_val)+1
##print "min error tree",error_index


##
##temp=x_arr[0]
##tempo=tree1[f1]
##
##label= comparison(temp,tempo,f1,f2,f3)
##print "Label generated is ",label






'''































print tree1

##
##
tempo=tree1[f1]
print tempo
##
##print tempo[f1]
##
##
##f1=5
##f2=2
##f3=3
##temp=x_arr[1]
##tempo=tree1[f1]
##comparison(temp,tempo,f1,f2,f3)
##        
##
##
##if isinstance(tempo[1],dict):
##    print "True"
'''    
