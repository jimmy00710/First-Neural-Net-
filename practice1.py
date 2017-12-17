import os 
import numpy as np 
import pandas as pd 

path = os.getcwd()
path = path + '/airfoil.txt'
data = pd.read_table(path,header = None,names = ['Frequency','Angle','Chord_Length','Velocity','Suction','Scaled_sound'])
#print(data.head())
data = (data - data.mean())/data.std()
print(data.head())

def computeCost(X,y,theta):
	hypothesis = np.power(((X*theta.T)-y),2)
	return np.sum(hypothesis)/(2*len(X))

# data.shape gives us dimension of our data in matrix format
# when we do cols = data.shape[1] gives us the column 

cols = data.shape[1]

X = data.iloc[:,0:cols - 1]
y = data.iloc[:,cols - 1:cols]
#print(y.head)
#print(y.shape)
#print(y.values) gives us values in array format 
# so let us convert our X and y in matrix 
#print(X.values)

X = np.matrix(X.values,dtype = np.int64)
y = np.matrix(y.values,dtype = np.int64)
print(X.shape)
print(y.shape)
theta = np.matrix(np.array([0,0,0,0,0]),dtype = np.float64)
print(theta.shape)
print(computeCost(X,y,theta)) #7815.78620446
#print(X.values)

def gradientDescent(X, y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

alpha = 0.0005
iters = 30000

g,cost = gradientDescent(X,y,theta,alpha,iters)
print(g)

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X,y)
print(reg.coef_)
print(computeCost(X,y,g))
#s = np.sum(np.power((y-(reg.predict(X))),2))
#print(s/2*len(X))
#import seaborn as sns 
#import matplotlib.pyplot as plt 
#sns.lm
import matplotlib.pyplot as plt 
fig,ax = plt.subplots(figsize = (12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs training Epoch')
plt.show()