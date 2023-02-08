import numpy as np
from sklearn import preprocessing
from pandas import read_csv
path=r"diabetes.csv"
data= read_csv(path)
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
array= data.values
#displaying the mean and the standard deviation of the input data
print("Mean =", array.mean(axis=0))
print("Stddeviation = ", array.std(axis=0))
#Removing the mean and the standard deviation of the input data

data_scaled = preprocessing.scale(array)
print("Mean_removed =\n", data_scaled.mean(axis=0))
print("\n\n")
print("Stddeviation_removed =\n", data_scaled.std(axis=0))