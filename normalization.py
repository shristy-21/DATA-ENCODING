from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer
path=r'diabetes.csv'
data= read_csv(path)
names=['preg','plas','skin','test','mass','pedi','age','class']
array=data.values
data_normalizer = Normalizer(norm ='max').fit(array)
data_normalized =data_normalizer.transform(array) 
print("\n L1 NORMALIZED DATA:\n  ",data_normalized[0:3])

