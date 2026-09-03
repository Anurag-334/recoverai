import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from app.ml.features import engineer_features

def train_model():
    data_path = "data/synthetic/recovery_events.csv"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Process features
    processed_df, features, encoders = engineer_features(df)
    
    X = processed_df[features]
    y = processed_df["recovered"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained. Test Accuracy: {accuracy:.4f}")
    
    # Save artifacts
    artifacts = {
        "model": model,
        "features": features,
        "encoders": encoders
    }
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(artifacts, "models/model.joblib")
    print("Model artifacts saved to models/model.joblib")

if __name__ == "__main__":
    train_model()
