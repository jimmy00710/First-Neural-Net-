import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

path  = os.getcwd()
path = path + '/energydata_complete.csv'
data = pd.read_csv(path , header = None)
data = data.iloc[2:]
print(data.head())