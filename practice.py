import numpy as np
import pandas as pd 
import category_encoders as ce
#READING THE TRAIN .CSV FILE
train_df=pd.read_csv('Data_Train.csv')
print(train_df)
# PRINTING THE DATATYPE
print(train_df.dtypes)
#SELECTING THE DATATYPE
cat=train_df.select_dtypes(include=['object']).copy()
cat.head
print(cat)
#CONVERTING THE DATATYPE TO CATEGORY
train_df["Airline"] = train_df["Airline"].astype('category')
print(train_df.dtypes)
train_df["Airline_cat"] = train_df["Airline"].cat.codes
train_df.head()
print(train_df)
# ONE HOT ENCODING TECHNIQUES
print("ONE HOT ENCODING TECHINIQUES \n")
encoder=ce.OneHotEncoder(cols='Airline',return_df=True,handle_unknown='retun_nan',use_cat_names=True)
data_encoder=encoder.fit_transform(train_df)
print(data_encoder)
#DUMMY ENCODING TECHIQUES
print("DUMMY ENCODING\n")
data_encoder=pd.get_dummies(train_df,drop_first=True)
print(data_encoder)
#BINARY ENCODING 
print("binary encoder\n")
encoder=ce.BinaryEncoder(cols=['Source'],return_df=True)
data_encoder=encoder.fit_transform(train_df)
print(data_encoder)
#BASE N ENCODING TECHIQUE
print("BASE N ENCODING....\n")
encoder= ce.BaseNEncoder(cols=['Source'],return_df=True,base=5)
data_encoder=encoder.fit_transform(train_df)
print(data_encoder)
#TARGET ENCODING TECHIQUE
print("TARGET ENCODER.....\n")
encoder=ce.TargetEncoder(cols='Airline') 
data_encoder=encoder.fit_transform(train_df['Airline'],train_df['Airline_cat'])
print(data_encoder)


'''#make binary of labels
for label in top10:
    new_df[label]=np.where(new_df['Source']==label,1,0)
    new_df[['Source']+top10]
print(new_df)'''

