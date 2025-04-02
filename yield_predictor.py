# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 21:09:51 2025

@author: pc
"""

import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
#from tensorflow.keras.losses import mean_squared_error
from keras.losses import mean_squared_error
import joblib

# Load the trained CNN-MLP model and scaler
try:
    rend_model = load_model("CNN-MLP.h5",
                            custom_objects={'mse': mean_squared_error})
    scaler = joblib.load("scaler_mlp.pkl") # Assuming you saved the scaler
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Please check the file paths.")
    st.stop()

# Title of the app
st.title("Yield Prediction using CNN-MLP")
st.markdown("Enter the reaction conditions and molecule SMILES to predict the yield.")

# Input fields for reaction conditions
st.subheader("Reaction Conditions")
time = st.number_input("Time (min)", min_value=0.0, max_value=910.0, value=120.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=200.0, value=100.0)
pressure = st.number_input("Pressure (MPa)", min_value=10.0, max_value=100.0, value=50.0)
cosolvent = st.number_input("CoSolvent(W/W)", min_value=0.0, max_value=1.0, value=0.0)
flow_rate = st.number_input("Flow Rate (g/min)", min_value=0.0, max_value=100.0, value=7.36)

# Input fields for molecule SMILES
st.subheader("Molecule SMILES")
mol1_smiles = st.text_input("SMILES for Molecule 1", "CCO")
mol2_smiles = st.text_input("SMILES for Molecule 2", "CCC")
mol3_smiles = st.text_input("SMILES for Molecule 3", "C=C")
mol4_smiles = st.text_input("SMILES for Molecule 4", "O=O")

# Function to generate Morgan Fingerprint
def generate_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_array = np.zeros((1, 1024), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, fp_array[0])
        return fp_array
    else:
        st.error(f"Invalid SMILES: {smiles}")
        return None

# Prediction button
if st.button("Predict Yield"):
    # Prepare input conditions
    conditions = np.array([[time, temperature, pressure, cosolvent, flow_rate]])
    scaled_conditions = scaler.transform(conditions)

    # Generate fingerprints for molecules
    fp1 = generate_fingerprint(mol1_smiles)
    fp2 = generate_fingerprint(mol2_smiles)
    fp3 = generate_fingerprint(mol3_smiles)
    fp4 = generate_fingerprint(mol4_smiles)

    if fp1 is not None and fp2 is not None and fp3 is not None and fp4 is not None:
        fingerprints = np.stack([fp1[0], fp2[0], fp3[0], fp4[0]], axis=-1)
        fingerprints = fingerprints.reshape(1, 32, 32, 4) # Reshape for CNN

        # Make prediction
        predicted_yield = rend_model.predict([scaled_conditions, fingerprints])[0][0]

        # Display the prediction
        st.subheader("Prediction Result:")
        st.success(f"The predicted yield is: {predicted_yield:.2f} %")