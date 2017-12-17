import os 
import  numpy as np 
import pandas as pd 
from sklearn import linear_model 

path = os.getcwd() + '/ex1data2.txt'
data = pd.read_csv(path,header = None,names = ['Size','Bedroom','Price'])
data = (data - data.mean())/data.std()
print(data.head())
print(data.describe())

def computeCost(X,y,theta):
	inner = np.power(((X*theta.T)-y),2) #theta.T means theta transpose
	return np.sum(inner)/(2*len(X))


data.insert(0,'Ones',1)

cols = data.shape[1]
X = data.iloc[:,0:cols - 1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))

print(computeCost(X,y,theta))


def gradientDescent(X,y,theta,alpha,iters):
	temp = np.matrix(np.zeros(theta.shape))
	parameters = int(theta.ravel().shape[1])
	cost = np.zeros(iters)

	for i in range(iters):
		error = (X*theta.T) - y

		for j in range(parameters):
			term = np.multiply(error,X[:,j])
			temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

		theta = temp
		cost[i] = computeCost(X,y,theta)

	return theta,cost

alpha = 0.009599
iters = 3000

#reg.fit(X,y)
#print(reg.predict(X))
g,cost = gradientDescent(X,y,theta,alpha,iters)
print(g)
print(computeCost(X,y,g))
