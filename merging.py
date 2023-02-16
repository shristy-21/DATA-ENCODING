from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
df=pd.concat(map(pd.read_csv,['Data_Train.csv','Test_set.csv']),ignore_index=True)
print(df)
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,Y_test=train_test_split(x,y,test_size=0.05,random_state=0)
print("X TEST\n",X_test)
print("X TRAIN\n",X_train)
print("Y TEST\n",Y_test)
print('Y TRAIN\n',y_train)
names=['Airline','Source']
array= df.values
binarizer =Binarizer(threshold=0.5).fit(array)
binarized= binarizer.transform(array)
print("\n BINARY DATA :\n ",binarized[0:5])


'''names=['Airline','Source']
array=df.values 
data_normalizer = Normalizer(norm ='max').fit(array)
data_normalized =data_normalizer.transform(array) 
print("\n L1 NORMALIZED DATA:\n  ",data_normalized[0:3])'''