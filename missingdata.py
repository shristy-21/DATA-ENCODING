import pandas as pd
import numpy as np

def fill_null_values():
   
    df = pd.DataFrame(eval(input("Enter the DataFrame in the format: "
                                  "pd.DataFrame({'col1': [val1, val2, ...], "
                                  "'col2': [val1, val2, ...], ...})")))
    
    if df.empty:
        return df
    
    for column in df.columns:
        if df[column].dtype in [np.int64, np.float64]:
            mean = df[column].mean()
            df[column].fillna(mean, inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
        
    return df
    
   
df_filled = fill_null_values()


print(df_filled)