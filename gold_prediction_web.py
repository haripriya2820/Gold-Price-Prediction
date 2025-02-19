# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:57:57 2025

@author: harip
"""

import numpy as np
import pickle
import streamlit as sp
import pandas as pd

loaded_model=pickle.load(open('D:/Downloads/trained_model.sav','rb'))

#creating a function for prediction
def gold_prediction(input_values):
    
    
    feature_names = ["SPX","USO","SLV","EUR/USD"]  # Replace with actual feature names

    num = np.asarray(input_values)
    reshape = num.reshape(1, -1)  # Reshape to (1, 4)

    # Convert to DataFrame with feature names
    df_input = pd.DataFrame(reshape, columns=feature_names)

    predictions = loaded_model.predict(df_input)
    return predictions


def main():
    
    #giving a title
    sp.title("Gold prediction Web App")
    
    #getting input from user

    
    spx=sp.text_input('SPX')
    uso=sp.text_input('USO')
    slv=sp.text_input('SLV')
    eu=sp.text_input('EUR/USD')
    
    #CODE for prediction 
    dig=''
    #create a button for prediction
    if sp.button("Predict gold price"):
        dig=gold_prediction([spx,uso,slv,eu])
        print("The predicted gold price is ")
        
    sp.success(dig)
    
if __name__=='__main__':
    main()
        
    
    