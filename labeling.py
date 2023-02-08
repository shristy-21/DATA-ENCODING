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
