# 🤖 ChurnGuard AI

**Real-time Customer Retention Analytics Dashboard**

ChurnGuard AI is a full-stack data science application designed to predict customer churn, diagnose risk factors, and prescribe retention strategies in real-time. Unlike standard static analysis, this tool allows users to simulate "What-If" scenarios (e.g., "What if we lower the price by $20?") to see the immediate impact on a customer's risk score.

## 🚀 Key Features

* **Real-Time Predictions:** Powered by an **XGBoost** machine learning model (~81% Accuracy).
* **Interactive Simulations:** Adjust tenure, charges, and contracts to see live risk updates.
* **Customer Database:** Integrated **SQLite** database to search and load real customer profiles.
* **Prescriptive Analytics:** AI-driven recommendation engine (e.g., *"Switching to a 1-year contract reduces risk by 15%"*).
* **Modern UI:** Glassmorphism design with Dark/Light mode toggle.

## 🛠️ Tech Stack

* **Frontend:** Python Dash, Plotly, Bootstrap
* **Backend:** Flask (via Dash), SQLite
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas
* **Deployment:** Render (Cloud)

## 📸 Screenshots




## 📦 How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/dRock-Codez/churn-guard.git](https://github.com/dRock-Codez/churn-guard.git)
    cd churn-guard
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**
    ```bash
    python app.py
    ```
    Open your browser to `http://127.0.0.1:8050/`