# prediction.py

import numpy as np

def predict_pcos(model, X_processed):
    """
    Predict PCOS risk and probability using trained model
    """

    # Binary prediction
    prediction = model.predict(X_processed)[0]

    # Probability (for PCOS class = 1)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_processed)[0][1]
    else:
        probability = None

    return prediction, probability
