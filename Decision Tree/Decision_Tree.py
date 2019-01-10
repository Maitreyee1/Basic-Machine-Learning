'''
Maitreyee Mhasakar: net ID: mam171630
Problem Set 2: 
Problem 1: 
Following is the solution for Problem 2 :
Decision Trees on Poisonous Mushroom Dataset

'''

from __future__ import division
import csv
import numpy as np
from math import log10

# READING DATA
with open('mush_train.data') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    dataset=[]
    x_arr=[]    #training data set
    y_arr=[]    #training Output label 1 or -1
    for row in csv_reader:
        dataset.append(row)
        dimension=len(row[1:]) #dimension of input data x
        
        x_arr.append(row[1:])
        y_arr.append(row[0])

print len(dataset)        
x_arr=np.array(x_arr)
y_arr=np.array(y_arr)
Feature_rec=[]
Category_rec=[]
Hy=0
ig=[]
pure=False

Mushroom_Features={0:'cap-shape',1:'cap-surface',2:'cap-color',3:'bruises',4:'odor',5:'gill-attachment',
6:'gill-spacing',  
7:'gill-size',
8:'gill-color', 
9:'stalk-shape',
10:'stalk-root', 
11:'stalk-surface-above-ring', 
12:'stalk-surface-below-ring', 
13:'stalk-color-above-ring',
14:'stalk-color-below-ring', 
15:'veil-type', 
16:'veil-color',
17:'ring-number', 
18:'ring-type',
19:'spore-print-color',
20:'population', 
21:'habitat',}







def Calculate_Y_Entropy(y_arr):
    total_samples=len(y_arr)
    y_arr=list(y_arr)
    #H(Y)= H(Edible'e')+H(Poisonous: 'p')
    e=y_arr.count('e')
    p_edible= y_arr.count('e')/total_samples
    #Probability of poisonous = Number of poisonous mushrooms in given y_arr/Total number of mushrooms in given y_arr
    e_edible= -1*(p_edible*log10(p_edible)) #Entropy of Edible mushrooms
    #Probability of edible= Number of edible mushrooms in given y_arr/Total number of mushrooms in given y_arr
    p_poisonous= y_arr.count('p')/total_samples
    e_poisonous= -1*(p_poisonous)*log10(p_poisonous)#entropy of poisonous 
    Hy=e_edible+e_poisonous
    return Hy #Y entropy

def X(x_arr,y_arr):

    Entropy_compare=[]

    for i in range(len(x_arr[0])):
        Feature=x_arr[:,i]
        category=set(Feature) #Categories in X feature
        #print type(category)
        category=list(category)
        #print type(category)
        #print len(category)

        f_entropy=x_feature_entropy(Feature,category,y_arr)
        Entropy_compare.append(f_entropy) #Array of entropies of all features
    return Entropy_compare
    
'''

Attribute Information:

1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s 
2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s 
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y 
4. bruises?: bruises=t,no=f 
5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s 
6. gill-attachment: attached=a,descending=d,free=f,notched=n 
7. gill-spacing: close=c,crowded=w,distant=d 
8. gill-size: broad=b,narrow=n 
9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y 
10. stalk-shape: enlarging=e,tapering=t 
11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? 
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s 
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s 
14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
16. veil-type: partial=p,universal=u 
17. veil-color: brown=n,orange=o,white=w,yellow=y 
18. ring-number: none=n,one=o,two=t 
19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z 
20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y 
21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y 
22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
'''




def x_feature_entropy(Xf,category,y):
# Calculate entropy of a feature Xf
    size=len(Xf)
    blank=[]
    counter=[0]*len(category)
    count_y=[]
    e=[0]*len(category)
    p=[0]*len(category)


    for i in range(size):
        temp=Xf[i]
        #print temp
        
        for j in range(len(category)):
            if temp==category[j]:
                #print category[j]
                counter[j]=counter[j]+1 # Counter is  an array that keeps record of number of data points of each category of a feature
                if y[i]=='e':
                    e[j]+=1 #e array keeps record of number of edible mushrooms of a certain category
                else:
                    p[j]+=1 #p array keeps record of number of poisonous mushrooms of a certain category

            elif temp=='':
                blank.append(i)


    # Calculate Entropy for each category

    #print "Counter:",counter
    #print "e",e
    #print "p",p
    #print "Blank",blank


    

    Summ=0
#H(Y|X1)=H(Y|X1=a)+H(Y|X1=b)+H()Y|X1=c)+H()Y|X1=d)...
    for i in range(len(counter)):
        prob= counter[i]/size #probability of category
        #print category[i],counter[i],prob
        #print "e",e[i]
        #print "p",p[i]

        if counter[i]==0:
            Entropy=0
        else:
            prob_e=e[i]/counter[i] # Conditional probability : probability label is e in when feature is Xf 
            prob_p=p[i]/counter[i] # Conditional probability : probability label is p in when feature is Xf
            if prob_e==0:
                v=0
            else:
                v=prob_e*(log10(prob_e)) #Entropy e

            if prob_p==0:
                w=0
            else:
                w=prob_p*(log10(prob_p))# entropy p
            
            Entropy= -1*prob*(v+w)
            Summ+=Entropy #H(X1)

    #H(Y|X1)= H(Y|X1=a)+H(Y|X1=b)+H(Y|X1=c)+....

    return Summ #H(Y|X1)


def information_gain(Hy,Hx):
    #Information gain= H(y)-H(X[i]) where X[i] is one feature.

    for i in range(len(Hx)):
        IG= Hy-Hx[i]
        ig.append(IG)

    
    #print "IG",IG
    
    #return IG


def max_info_gain():

    Maximum=0
    Feature_num=0

    for i in range(len(ig)):
        if ig[i]> Maximum:
            Maximum=ig[i]
            Feature_num=i

    Feature_rec.append(Feature_num)
    return Maximum,Feature_num # Select Feature Num having max info gain Maximum 

        

##
##    
##    maximum,=max(ig[0]),
##    feature=ig.index(maximum)
##    print "Max",maximum
##



def Build_tree(x_arr,_y_arr,Feature_num):
    f_cat_map={}
    for i in range(dimension):
        temp=x_arr[:,i]
        f_cat_map[i]=list(set(temp)) #Stores features and its distinct categories

    tree[Feature_num]={}


    #Split tree based on feature Feature num

    c_dict={}

    for i in f_cat_map[Feature_num]:
        c_dict[i]=[]

    print "C_dict",c_dict #stores each category and correspondind sub tree
        



    for i in range(len(x_arr)):
        arr=[]
        temp_x=x_arr[i]
        label=y_arr[i]
        for j in range(len(f_cat_map[Feature_num])):
            if temp_x[Feature_num]==f_cat_map[Feature_num][j]:
                c_dict[temp_x[Feature_num]].append((i,x_arr[i],y_arr[i])) # C_dict key will have category, and its value will be its subtree  

    arr_cat=[]
    for i in c_dict:
        temp=[]
        

        for j in c_dict[i]:
            temp.append(j[2])
        print "Count of e",temp.count('e') # Number of edible in subtree of category
        print "Count of p",temp.count('p')# Number of poisonous in subtree of category

        if temp.count('e')==len(temp): #if all labels are e, prune the tree; data is pure terminate the branch
            c_dict[i]=True
            print i,"is e"
        elif temp.count('p')==len(temp): #if all labels are p, prune the tree data is pure terminate the branch
            c_dict[i]=True
            print i,"is p"
        else:
            arr_cat.append(i)
            print i, "is mixed" #build tree further on this subtree
    #print arr_cat


    if len(arr_cat)==0:
        print "Tree ready" # Entire Subtreehas pure data ; stopbuilding the tree
    else:
        sub_tree={} # Subtree of one entire feature
        for i in arr_cat:
            sub_tree[i]=c_dict[i]
            
                
#        print tree

        tree[Feature_num]=sub_tree # Append the tree to the original decision tree
    #print tree
    return sub_tree


##x_arr=[]
##y_arr=[]
##for i in sub_tree:
##    temporary=sub_tree[i]
##    for j in temporary:
##        x_arr.append(list(j[1]))
##        y_arr.append(j[2])
###print x_arr
###print y_arr
##    
















#Feature 1 Max info gain

Hy=Calculate_Y_Entropy(y_arr)
Hx=X(x_arr,y_arr)
information_gain(Hy,Hx)
#print ig
#print len(ig)
max_ig,Feature_num=max_info_gain() #Feature to split data on
print "maximum Info gain: ",max_ig,"At feature ",Feature_num
tree={}

#Feature 1 Tree building


sub_tree=Build_tree(x_arr,y_arr,Feature_num)
#print sub_tree

for i in sub_tree:
    x_arr=[]
    y_arr=[]

    temporary=sub_tree[i]
    for j in temporary:
        x_arr.append(list(j[1])) #New X formed by original X partitioned on feature
        y_arr.append(j[2]) #New Y formed by original y partitioned on feature
    x_arr=np.array(x_arr)
    y_arr=np.array(y_arr)
    Hy=0
    ig=[]

    
#print x_arr
#print y_arr
##
Hy=Calculate_Y_Entropy(y_arr)
Hx=X(x_arr,y_arr)
information_gain(Hy,Hx)
print ig
###print len(ig)
max_ig,Feature_num=max_info_gain()
print "maximum Info gain: ",max_ig,"At feature ",Feature_num #feature 2
##

#Build_tree() mechanism


f_cat_map={}
for i in range(dimension):
    temp=x_arr[:,i]
    f_cat_map[i]=list(set(temp)) #Stores features and its distinct categories

tree[Feature_num]={}


#Split tree based on feature Feature num

c_dict={}

for i in f_cat_map[Feature_num]:
    c_dict[i]=[]

print "C_dict",c_dict #stores each category and correspondind sub tree
    

array=[[None]]*len(f_cat_map[Feature_num])

for i in range(len(x_arr)):
    arr=[]
    temp_x=x_arr[i]
    label=y_arr[i]
    for j in range(len(f_cat_map[Feature_num])):
        if temp_x[Feature_num]==f_cat_map[Feature_num][j]:
            c_dict[temp_x[Feature_num]].append((i,x_arr[i],y_arr[i]))

arr_cat=[]
for i in c_dict:
    temp=[]
    

    for j in c_dict[i]:
        temp.append(j[2])
    print "Count of e",temp.count('e')
    print "Count of p",temp.count('p')

    if temp.count('e')==len(temp):
        c_dict[i]=True
        print i,"is e"
    elif temp.count('p')==len(temp):
        c_dict[i]=True
        print i,"is p"
    else:
        arr_cat.append(i)
        print i, "is mixed"
#print arr_cat


if len(arr_cat)==0:
    print "Tree ready"
    pure=True
else:
    sub_tree={}
    for i in arr_cat:
        sub_tree[i]=c_dict[i]
        
            
    #print tree

    tree[Feature_num]=sub_tree
#    print tree


x_arr=[]
y_arr=[]
for i in sub_tree:
    print "i in spore",i
    temporary=sub_tree[i]
    for j in temporary:
        x_arr.append(list(j[1]))
        y_arr.append(j[2])
#print x_arr
#print y_arr
#print tree
##
##
##
##
###Feature 3
x_arr=np.array(x_arr)
y_arr=np.array(y_arr)
Hy=0
ig=[]





print y_arr[0]

Hy=Calculate_Y_Entropy(y_arr)
Hx=X(x_arr,y_arr)
information_gain(Hy,Hx)
print ig
###print len(ig)
max_ig,Feature_num=max_info_gain()
print "maximum Info gain: ",max_ig,"At feature ",Feature_num
##
f_cat_map={}
for i in range(dimension):
    temp=x_arr[:,i]
    f_cat_map[i]=list(set(temp)) #Stores features and its distinct categories

tree[Feature_num]={}


#Split tree based on feature Feature num

c_dict={}

for i in f_cat_map[Feature_num]:
    c_dict[i]=[]

print "C_dict",c_dict #stores each category and correspondind sub tree
    

array=[[None]]*len(f_cat_map[Feature_num])

for i in range(len(x_arr)):
    arr=[]
    temp_x=x_arr[i]
    label=y_arr[i]
    for j in range(len(f_cat_map[Feature_num])):
        if temp_x[Feature_num]==f_cat_map[Feature_num][j]:
            c_dict[temp_x[Feature_num]].append((i,x_arr[i],y_arr[i]))

arr_cat=[]
for i in c_dict:
    temp=[]
    

    for j in c_dict[i]:
        temp.append(j[2])
    print "Count of e",temp.count('e')
    print "Count of p",temp.count('p')

    if temp.count('e')==len(temp):
        c_dict[i]=True
        print i,"is e"
    elif temp.count('p')==len(temp):
        c_dict[i]=True
        print i,"is p"
    else:
        arr_cat.append(i)
        print i, "is mixed"
#print arr_cat


if len(arr_cat)==0:
    print "Tree ready"
else:
    sub_tree={}
    for i in arr_cat:
        sub_tree[i]=c_dict[i]
        
            
    #print tree

    tree[Feature_num]=sub_tree
#    print tree
for i in tree:
    print i





for i in sub_tree:
    print i
    x_arr=[]
    y_arr=[]
    if i=='l':
        temporary=sub_tree[i]
        for j in temporary:
            x_arr.append(list(j[1]))
            y_arr.append(j[2])

        x_arr=np.array(x_arr)
        y_arr=np.array(y_arr)
        Hy=0
        ig=[]
        print y_arr[0]
        
        print x_arr

        Hy=Calculate_Y_Entropy(y_arr)
        Hx=X(x_arr,y_arr)
        information_gain(Hy,Hx)
        print ig
##        ###print len(ig)
        max_ig,Feature_num=max_info_gain()
        print "maximum Info gain: ",max_ig,"At feature ",Feature_num

        f_cat_map={}
        for i in range(dimension):
            temp=x_arr[:,i]
            f_cat_map[i]=list(set(temp)) #Stores features and its distinct categories

        tree[Feature_num]={}


        #Split tree based on feature Feature num

        c_dict={}

        for i in f_cat_map[Feature_num]:
            c_dict[i]=[]

        print "C_dict",c_dict #stores each category and correspondind sub tree
            

        #array=[[None]]*len(f_cat_map[Feature_num])

        for i in range(len(x_arr)):
            arr=[]
            temp_x=x_arr[i]
            label=y_arr[i]
            for j in range(len(f_cat_map[Feature_num])):
                if temp_x[Feature_num]==f_cat_map[Feature_num][j]:
                    c_dict[temp_x[Feature_num]].append((i,x_arr[i],y_arr[i]))

        arr_cat=[]
        for i in c_dict:
            temp=[]
            

            for j in c_dict[i]:
                temp.append(j[2])
            print "Count of e",temp.count('e')
            print "Count of p",temp.count('p')

            if temp.count('e')==len(temp):
                c_dict[i]=True
                print i,"is e"
            elif temp.count('p')==len(temp):
                c_dict[i]=True
                print i,"is p"
            else:
                arr_cat.append(i)
                print i, "is mixed"
        #print arr_cat


        if len(arr_cat)==0:
            print "Tree ready"
        else:
            sub_tree={}
            for i in arr_cat:
                sub_tree[i]=c_dict[i]
        print sub_tree


    elif i=='d':
        temporary=sub_tree[i]
        for j in temporary:
            x_arr.append(list(j[1]))
            y_arr.append(j[2])

        x_arr=np.array(x_arr)
        y_arr=np.array(y_arr)
        Hy=0
        ig=[]
        print y_arr[0]
        
        print x_arr

        Hy=Calculate_Y_Entropy(y_arr)
        Hx=X(x_arr,y_arr)
        information_gain(Hy,Hx)
        print ig
##        ###print len(ig)
        max_ig,Feature_num=max_info_gain()
        print "maximum Info gain: ",max_ig,"At feature ",Feature_num

        f_cat_map={}
        for i in range(dimension):
            temp=x_arr[:,i]
            f_cat_map[i]=list(set(temp)) #Stores features and its distinct categories

        tree[Feature_num]={}


        #Split tree based on feature Feature num

        c_dict={}

        for i in f_cat_map[Feature_num]:
            c_dict[i]=[]

        print "C_dict",c_dict #stores each category and correspondind sub tree
            

        #array=[[None]]*len(f_cat_map[Feature_num])

        for i in range(len(x_arr)):
            arr=[]
            temp_x=x_arr[i]
            label=y_arr[i]
            for j in range(len(f_cat_map[Feature_num])):
                if temp_x[Feature_num]==f_cat_map[Feature_num][j]:
                    c_dict[temp_x[Feature_num]].append((i,x_arr[i],y_arr[i]))

        arr_cat=[]
        for i in c_dict:
            temp=[]
            

            for j in c_dict[i]:
                temp.append(j[2])
            print "Count of e",temp.count('e')
            print "Count of p",temp.count('p')

            if temp.count('e')==len(temp):
                c_dict[i]=True
                print i,"is e"
            elif temp.count('p')==len(temp):
                c_dict[i]=True
                print i,"is p"
            else:
                arr_cat.append(i)
                print i, "is mixed"
        #print arr_cat


        if len(arr_cat)==0:
            print "Tree ready"
        else:
            sub_tree={}
            for i in arr_cat:
                sub_tree[i]=c_dict[i]
        print sub_tree


tree[Feature_num]=sub_tree





##print tree
##for i in tree:
##    print "i :"
##    for j in sub_tree:
##        print "j"
##    
print "Selected Features:"
for i in Feature_rec:
    print i,Mushroom_Features[i]

def Test_Accuracy():
        
    # READING DATA
    with open('mush_test.data') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        dataset=[]
        for row in csv_reader:
            dataset.append(row)
    size=len(dataset)

        
    '''

    Attribute Information:

    1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s 
    2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s 
    3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y 
    4. bruises?: bruises=t,no=f 
    5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s 
    6. gill-attachment: attached=a,descending=d,free=f,notched=n 
    7. gill-spacing: close=c,crowded=w,distant=d 
    8. gill-size: broad=b,narrow=n 
    9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y 
    10. stalk-shape: enlarging=e,tapering=t 
    11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? 
    12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s 
    13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s 
    14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
    15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
    16. veil-type: partial=p,universal=u 
    17. veil-color: brown=n,orange=o,white=w,yellow=y 
    18. ring-number: none=n,one=o,two=t 
    19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z 
    20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y 
    21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y 
    22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

    '''
    Features=[5, 20, 22, 3, 8]

    correct=0
    misclassify=0
    counte=0
    countp=0


    for i in range(len(dataset)): # if data contains seletced combination of features, it predicts a label based on the decision tree
        temp=dataset[i]
        if temp[5]=='a':
          
            if temp[0]=='e':
                counte+=1
                correct+=1 #if predicted label==actual label in test, increment count of correctly classified data
            else:
                misclassify+=1

        if temp[5]=='c':
            if temp[0]=='p':
                countp+=1
                correct+=1
            else:
                misclassify+=1

        if temp[5]=='f':
            if temp[0]=='p':
                countp+=1
                correct+=1
            else:
                misclassify+=1

        if temp[5]=='m':
            if temp[0]=='p':
                countp+=1
                correct+=1
            else:
                misclassify+=1

        if temp[5]=='l':
            if temp[0]=='e':
                counte+=1
                correct+=1
            else:
                misclassify+=1

        if temp[5]=='n':

            if temp[20]=='k':
                if temp[0]=='e':
                    counte+=1
                    correct+=1
                else:
                    misclassify+=1
            if temp[20]=='n':

                if temp[0]=='e':
                    counte+=1
                    correct+=1
                else:
                    misclassify+=1
            if temp[0]=='b':
                if temp[0]=='e':
                    counte+=1
                    correct+=1

                else:
                    misclassify+=1

            if temp[20]=='h':
                if temp[0]=='e':
                    counte+=1
                    correct+=1
                else:
                    misclassify+=1

            if temp[20]=='r':
                if temp[0]=='p':
                    countp+=1
                    correct+=1
                else:
                    misclassify+=1

            if temp[20]=='o':
                if temp[0]=='e':
                    counte+=1
                    correct+=1
                else:
                    misclassify+=1


            if temp[20]=='w':
                    if temp[22]=='p':
                        if temp[0]=='e':
                            counte+=1
                            correct+=1
                        else:
                            misclassify+=1
                    if temp[22]=='w':
                        if temp[0]=='e':
                            counte+=1
                            correct+=1
                        else:
                            misclassify+=1
                    if temp[22]=='l':
                        if temp[3]=='u':
                            if temp[0]=='p':
                                countp+=1
                                correct+=1
                            else:
                                misclassify+=1
                        
                        if temp[3]=='c':
                            if temp[0]=='e':
                                counte+=1
                                correct+=1
                            else:
                                misclassify+=1

                        if temp[3]=='w':
                            if temp[0]=='p':
                                countp+=1
                                correct+=1
                            else:
                                misclassify+=1

                        if temp[3]=='n':
                            if temp[0]=='e':
                                counte+=1
                                correct+=1
                            else:
                                misclassify+=1


                        if temp[3]=='b':
                            if temp[0]=='e':
                                counte+=1
                                correct+=1
                            else:
                                misclassify+=1
                        if temp[3]=='g':
                            if temp[0]=='e':
                                counte+=1
                                correct+=1
                            else:
                                misclassify+=1
                        if temp[3]=='r':
                            if temp[0]=='e':
                                counte+=1
                                correct+=1
                            else:
                                misclassify+=1
                        if temp[3]=='p':
                            if temp[0]=='e':
                                counte+=1
                                correct+=1
                            else:
                                misclassify+=1
                        if temp[3]=='e':
                            if temp[0]=='e':
                                counte+=1
                                correct+=1
                            else:
                                misclassify+=1
                        if temp[3]=='y':
                            if temp[0]=='e':
                                counte+=1
                                correct+=1
                            else:
                                misclassify+=1
                        

                    if temp[22]=='g':
                        if temp[0]=='e':
                            counte+=1
                            correct+=1
                        else:
                            misclassify+=1
                    if temp[22]=='d':
                        if temp[8]=='b':
                            if temp[0]=='e':
                                counte+=1
                                correct+=1
                            else:
                                misclassify+=1
                        if temp[8]=='n':
                            if temp[0]=='p':
                                countp+=1
                                correct+=1
                            else:
                                misclassify+=1
                    if temp[22]=='m':
                        if temp[0]=='e':
                            counte+=1
                            correct+=1
                        else:
                            misclassify+=1
                    if temp[22]=='u':
                        if temp[0]=='e':
                            counte+=1
                            correct+=1
                        else:
                            misclassify+=1




            if temp[20]=='y':
                if temp[0]=='e':
                    counte+=1
                    correct+=1
                else:
                    misclassify+=1


        if temp[5]=='p':
            if temp[0]=='p':
                correct+=1
            else:
                misclassify+=1
        if temp[5]=='s':
            if temp[0]=='p':
                correct+=1
            else:
                misclassify+=1

        if temp[5]=='y':
            if temp[0]=='p':
                correct+=1
            else:
                misclassify+=1

#    print correct,misclassify
    Accuracy= (correct/size)*100 # Accuracy= (correctly classified points/ total number of points)*100
    print "Accuracy : ",Accuracy

Test_Accuracy()


































