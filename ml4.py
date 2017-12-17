import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat 

data = loadmat('ex3data1.mat')
print(data)
print(data['X'].shape)

def sigmoid(z):
	return 1/(1+np.exp(-z))

def cost(theta,X,y,learningRate):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
	second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
	reg = (learningRate/2*len(X)) * np.sum(np.power(theta(:,1:theta.shape[1])))
	return np.sum(first - second) / (len(X)) + reg


def gradient_with_loop(theta,X,y,learningRate):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)

	parameters = int(theta.ravel().shape[1])
	grad = np.zeros(parameters)

	error = sigmoid(X*theta.T) - y

	for i in range(parameters):
		term = np.multiply(error,X[:,i])

		if (i==0):
			grad[i] = np.sum(term)/len(X)
		else:
			grad[i] = (np.sum(term)/len(X)) + ((learningRate/len(X))*theta[:,i])

	return grad




def gradient(theta,X,y,learningRate):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)

	parameters = int