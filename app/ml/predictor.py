import joblib
import os
import numpy as np
from app.ml.features import preprocess_single

class RecoveryPredictor:
    def __init__(self, model_path="models/model.joblib"):
        self.model = None
        self.encoders = None
        self.features = None
        
        if os.path.exists(model_path):
            artifacts = joblib.load(model_path)
            self.model = artifacts["model"]
            self.encoders = artifacts["encoders"]
            self.features = artifacts["features"]
            
    def predict_probability(self, context_dict: dict) -> float:
        if not self.model:
            return 0.5 # fallback
            
        features_array = preprocess_single(context_dict, self.encoders)
        X = np.array(features_array).reshape(1, -1)
        
        prob = self.model.predict_proba(X)[0][1] # Probability of class 1 (recovered)
        return float(prob)
        
predictor = RecoveryPredictor()
