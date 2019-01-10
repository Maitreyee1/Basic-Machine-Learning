
#Maitreyee Mhasakar mam171630
#Problem Set 6 : Problem 1 
#Markov decision process 


REWARD = [[ 0, 1, 0.5,0.5 ],[ 0.5, 0, 1, 0.5], [ -1, 0.5, 0, 0.5 ], [-1,0.5,0.5,0]]
GAMMA = 0.8;
EPSILON = 10E-9;


#initialize the value function to zeros
valueFunction = []
valueFunction.append([0.0]*len(REWARD))
valueFunction.append([0.0]*len(REWARD))
valueFunction.append([0.0]*len(REWARD))
valueFunction.append([0.0]*len(REWARD))
citer = True

#value iteration
while citer==True:
    citer = False
    Vt=[]
    Vt=valueFunction[-1]
    Vtplus1 = [0.0]*len(Vt) #******
    for i in range(len(Vtplus1)-1):
        v = -999.0
        reward = REWARD[i]

        for j in range(len(reward)):
            newV = reward[j] + GAMMA * Vt[j]
            if newV > v:
                v = newV
        Vtplus1[i] = v
    valueFunction.append(Vtplus1)
    for i in range (len(Vt)):
        if abs(Vt[i] - Vtplus1[i]) > EPSILON :
            citer = True

		
#print value function
print(valueFunction)

#find optimal policy
opt_val = valueFunction[-1]
opt_pol = {}

for i in range(len(REWARD)):
    reward=[]
    reward = REWARD[i]
    Fval = -999.0
    fact = 0
    for j in range(len(reward)):
        if (reward[j] + opt_val[j]) > Fval:
            Fval = reward[j] + GAMMA * opt_val[j]
            fact = j
    f=fact + 1
    opt_pol[i + 1]=f

print(opt_pol)
