# model_loader.py

import joblib

MODEL_PATH = "artifacts/pcos_knn_model.pkl"
SCALER_PATH = "artifacts/pcos_scaler.pkl"
FEATURES_PATH = "artifacts/pcos_features.pkl"

def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features
