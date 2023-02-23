from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
import pandas as pd
df= pd.read_csv('diabetes.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,Y_test=train_test_split(x,y,test_size=0.05,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
classifier_knn = KNeighborsClassifier(n_neighbors = 3)
classifier_knn.fit(X_train, y_train)
y_pred = classifier_knn.predict(X_test)
# Finding accuracy by comparing actual response values(y_test)with predicted response value(y_pred)
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
