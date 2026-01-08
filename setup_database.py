import sqlite3
import pandas as pd
import os

db_name = "churn.db"
csv_name = "churn_data.csv"

# 1. Check if DB already exists (to avoid duplicates)
if os.path.exists(db_name):
    os.remove(db_name)
    print(f"🗑️ Removed old {db_name} to start fresh.")

# 2. Load the CSV
print(f"📂 Loading {csv_name}...")
try:
    df = pd.read_csv(csv_name)
except FileNotFoundError:
    print("❌ Error: CSV file not found!")
    exit()

# 3. Create/Connect to SQLite Database
print(f"🔌 Connecting to {db_name}...")
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# 4. Push Data to SQL
# This creates a table named 'customers' and dumps the dataframe into it
print("🚀 Migrating data to SQL table 'customers'...")
df.to_sql('customers', conn, if_exists='replace', index=False)

# 5. Verify it worked
cursor.execute("SELECT count(*) FROM customers")
row_count = cursor.fetchone()[0]

print(f"✅ Success! Database created with {row_count} customer records.")
print(f"💾 File saved as: {db_name}")

conn.close()