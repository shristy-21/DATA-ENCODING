#import modules:
import numpy as np
import pandas as pd 
import category_encoders as ce
# LOADING FILES 
print('LOADING FILES......')
train_df=pd.read_csv('Data_Train.csv')
cat=train_df.select_dtypes(include='object').copy()
print('DTYPE OF THE DATA.....')
print(cat)
# selecting top 10 from file
print(' SELECTING TOP 10 FROM THE FILE')
new_df=pd.read_csv('Data_Train.csv',usecols=['Airline','Date_of_Journey','Source'])
new_df=train_df.head()
print(new_df)
for x in new_df.columns:
    print(x,':',len(new_df[x].unique()))
#finding the top 10 categories
print('FINDING TOP 10 CATEGORY.... ')
new_df.Source.value_counts().sort_values(ascending=False).head(10)
print(new_df)
top10=[x for x in new_df.Source.value_counts().sort_values(ascending=False).head(10).index]
print(' TOP 10....')
print(top10)
#performimg one hot encoding at 'source ' col:
print(' PERFORMIMG ONE HOT ENCODING USING TOP 10 ....')
encoders=ce.OneHotEncoder(cols='Source',handle_unknown="return_nan",return_df=True,use_cat_names=True)
print(new_df)
data_encoder=encoders.fit_transform(new_df)
print(data_encoder)

