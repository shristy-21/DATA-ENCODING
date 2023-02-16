import pandas as pd
import numpy as np

def missing_values():
   
    data = pd.DataFrame(eval(input("Enter the DataFrame in the format: "
                                  "pd.DataFrame({'col1': [val1, val2, ...], "
                                  "'col2': [val1, val2, ...], ...})")))
    
    if data.empty:
        return data
    
    for column in data.columns:
        if data[column].dtype in [np.int64, np.float64]:
            mean = data[column].mean()
            data[column].fillna(mean, inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)
        
    return data
    
   
data_filled = missing_values()


print(data_filled)
