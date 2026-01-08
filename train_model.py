import sqlite3
import json
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load Data from SQL Database (Enterprise Style)
print("🔌 Connecting to SQL Database...")
try:
    conn = sqlite3.connect('churn.db')
    # We read directly from the 'customers' table we just created
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    print("✅ Data loaded from SQL successfully.")
except Exception as e:
    print(f"❌ Database Error: {e}")
    print("Did you run 'setup_database.py'?")
    exit()

# 2. Data Cleaning
# Handle TotalCharges blanks
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. Preprocessing
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Prepare X and y
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# 5. Train the Model
print("Training XGBoost on Real Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# UPDATED: Removed 'use_label_encoder' to fix the warning
model = xgb.XGBClassifier(
    eval_metric='logloss',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1
)
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"✅ Model Trained! Accuracy: {acc*100:.2f}%")

# 7. Save Model (THE FIX)
# We save the underlying 'booster' (the brain) to avoid the sklearn error
model.get_booster().save_model('model.json')

# 8. Save Feature Names
model_features = list(X_train.columns)
with open('features.json', 'w') as f:
    json.dump(model_features, f)

print("✅ Safety Lock: 'features.json' saved.")