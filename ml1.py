import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import os 
from sklearn import linear_model

reg = linear_model.LinearRegression()
path = os.getcwd()
path = path + '/ex1data.txt'
data = pd.read_csv(path,header = None,names = ['Population','Profit'])
#print(data.head())
#print(data.describe())

def computeCost(X,y,theta):
	inner = np.power(((X*theta.T)-y),2) #theta.T means theta transpose
	return np.sum(inner)/(2*len(X))


data.insert(0,'Ones',1)

cols = data.shape[1]
X = data.iloc[:,0:cols - 1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

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
alpha = 0.009599
iters = 3000

reg.fit(X,y)
#print(reg.predict(X))
g,cost = gradientDescent(X,y,theta,alpha,iters)
#you have to perform normalization 
# thats why error is coming 
print(g)
print(computeCost(X,y,g))
#print(reg.coef_)
