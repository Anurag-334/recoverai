import pandas as pd
from sklearn.preprocessing import LabelEncoder

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Process categorical features and return train ready df"""
    df = df.copy()
    
    categorical_cols = ["payment_method", "event_type", "failure_reason", "customer_segment"]
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    features = [
        "amount", "payment_method", "event_type", "failure_reason", 
        "attempt_number", "customer_segment", "days_since_failure", 
        "previous_success_rate", "customer_lifetime_value", 
        "is_subscription", "invoice_age"
    ]
    
    return df, features, encoders

def preprocess_single(data: dict, encoders: dict) -> list[float]:
    features = []
    
    amount = data.get("amount", 0.0)
    features.append(float(amount))
    
    for col in ["payment_method", "event_type", "failure_reason"]:
        val = data.get(col, "")
        try:
            encoded = encoders[col].transform([val])[0]
        except ValueError:
            # handle unknown categories
            encoded = -1
        features.append(float(encoded))
        
    features.append(float(data.get("attempt_number", 1)))
    
    val = data.get("customer_segment", "")
    try:
        encoded = encoders["customer_segment"].transform([val])[0]
    except ValueError:
        encoded = -1
    features.append(float(encoded))
    
    features.append(float(data.get("days_since_failure", 0)))
    features.append(float(data.get("previous_success_rate", 0.0)))
    features.append(float(data.get("customer_lifetime_value", 0.0)))
    features.append(float(data.get("is_subscription", 0)))
    features.append(float(data.get("invoice_age", 0)))
    
    return features
