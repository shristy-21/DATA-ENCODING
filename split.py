from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv('diabetes.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=0)
print("X TRAIN",X_train)
print('Y TRAIN',y_train)
clf = RandomForestClassifier(n_estimators = 100)  
clf.fit(X_train,Y_train)
  
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
from sklearn import metrics  
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
