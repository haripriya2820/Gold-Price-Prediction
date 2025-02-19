# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:41:07 2025

@author: harip
"""

import numpy as np
import pandas as pd
import pickle
#loading the trained model
loaded_model=pickle.load(open('D:/Downloads/trained_model.sav','rb'))
input_values = (1200.50, 65.30, 12.75, 1.25)  # New input values
feature_names = ["SPX","USO","SLV","EUR/USD"]  # Replace with actual feature names

num = np.asarray(input_values)
reshape = num.reshape(1, -1)  # Reshape to (1, 4)

# Convert to DataFrame with feature names
df_input = pd.DataFrame(reshape, columns=feature_names)

predictions = loaded_model.predict(df_input)
print("the predicted gold value is:",predictions)