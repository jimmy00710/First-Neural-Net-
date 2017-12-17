import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.io import loadmat

data = loadmat('ex8_movies.mat')
print(data)
Y = data['Y']
R = data['R']
print(Y.shape,R.shape)

print(Y[1,R[1,:]].mean())
