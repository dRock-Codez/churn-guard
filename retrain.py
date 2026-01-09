import sqlite3
import pandas as pd
import json
import traceback
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Global settings
DB_PATH = 'churn.db' 
MODEL_PATH = 'model.json'
FEATURES_PATH = 'features.json'

def run_retraining():
    try:
        # 1. Load Data
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM customers", conn)
        conn.close()
        
        # 2. Preprocessing
        # --- FIX: Force TotalCharges to numeric (handles empty strings " ") ---
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        # Drop ID and Target
        cols_to_drop = ['customerID', 'Churn']
        X = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
        y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

        # One-Hot Encoding
        X = pd.get_dummies(X, columns=['Contract'], drop_first=False)
        
        # Keep only numeric columns
        X = X.select_dtypes(include=['number', 'bool'])
        
        # Save feature names order (CRITICAL)
        feature_names = X.columns.tolist()
        with open(FEATURES_PATH, 'w') as f:
            json.dump(feature_names, f)

        # 3. Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Using the Sklearn wrapper for training is fine, 
        # but we extract the BOOSTER at the end.
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=100
        )
        model.fit(X_train, y_train)

        # 4. Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # 5. Save Raw Booster (Prevents metadata errors)
        model.get_booster().save_model(MODEL_PATH)

        return f"✅ Retrain Success! Accuracy: {acc*100:.1f}%"

    except Exception as e:
        print(traceback.format_exc()) 
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    print(run_retraining())