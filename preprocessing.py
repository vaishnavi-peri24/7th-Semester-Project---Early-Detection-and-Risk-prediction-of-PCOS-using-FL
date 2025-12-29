# preprocessing.py

import numpy as np
import pandas as pd

def preprocess_input(input_dict, feature_names, scaler):
    """
    Converts raw user input into a scaled model-ready dataframe
    """

    df = pd.DataFrame([input_dict])

    # Encode binary categorical values
    binary_map = {
        "Yes": 1, "No": 0,
        "Regular": 0, "Irregular": 1
    }

    for col in df.columns:
        df[col] = df[col].replace(binary_map)

    # Ensure all required features exist
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0

    # Reorder columns
    df = df[feature_names]

    # Scale features
    X_scaled = scaler.transform(df)

    return X_scaled
