from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
df=pd.read_csv('diabetes.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,Y_test=train_test_split(x,y,test_size=0.05,random_state=0)
print("X TRAIN",X_train)
print('Y TRAIN',y_train)