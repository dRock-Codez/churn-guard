import sqlite3

def create_history_table():
    conn = sqlite3.connect('churn.db')
    cursor = conn.cursor()
    
    # Create a simple history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            risk_score REAL,
            tenure INTEGER,
            monthly_charges REAL,
            contract TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ History table created successfully!")

if __name__ == "__main__":
    create_history_table()