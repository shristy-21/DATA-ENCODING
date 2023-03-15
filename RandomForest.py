import pandas as pd
import numpy as np

def fill_null_values():
   
    '''df = pd.DataFrame(eval(input("Enter the DataFrame in the format: "
                                  "pd.DataFrame({'col1': [val1, val2, ...], "
                                  "'col2': [val1, val2, ...], ...})")))'''
    df=pd.read_csv('Diabetes.csv')
    if df.empty:
        return df
    
    for column in df.columns:
        if df[column].dtype in [np.int64, np.float64]:
            mean = df[column].mean()
            df[column].fillna(mean, inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
        
    return df
    
print("FILLING UP  THE MISSING VALUES ")
df_filled = fill_null_values()


print(df_filled)
def dataencoding():
    import numpy as np
    from sklearn import preprocessing
    input_labels= ['justic','patience','justic','truth','patience','loyalty','perserverance']
    encoder=preprocessing.LabelEncoder()
    encoder.fit(input_labels)
    test_labels=['truth','justic','patience','loyalty']
    encoder_values=encoder.transform(test_labels)
    print('\n LABELS:',test_labels)
    print("\n ENCODED VALUES:",list(encoder_values))
    encoded_values= [2,0,4,3]
    decoded_values= encoder.inverse_transform(encoded_values)
    print("\n ENCODED VALUES:\n", encoded_values)
    print("\n DECODED VALUES :\n",list(decoded_values))

data=dataencoding()
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
import pandas as pd
df= pd.read_csv('Diabetes.csv')
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