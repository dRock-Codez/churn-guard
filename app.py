import retrain
import sqlite3
import json
import dash
import random
import xgboost as xgb
from dash import dcc, html, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

# --- 1. SETUP & LOAD RESOURCES ---
model = xgb.Booster()

def load_model_resources():
    global model, feature_names
    if os.path.exists('model.json'):
        model.load_model('model.json')
    try:
        with open('features.json', 'r') as f:
            feature_names = json.load(f)
    except:
        feature_names = []

load_model_resources()

''' def get_all_ids():
    try:
        conn = sqlite3.connect('churn.db')
        # LIMIT 2000 prevents the MemoryError by not loading too much data at once
        ids = pd.read_sql("SELECT customerID FROM customers LIMIT 2000", conn)['customerID'].tolist()
        conn.close()
        return ids
    except:
        return []

all_customer_ids = get_all_ids() '''

def safe_predict(input_df):
    if not feature_names: return 0.5
    for col in feature_names:
        if col not in input_df.columns: input_df[col] = 0
    input_df = input_df[feature_names]
    dmat = xgb.DMatrix(input_df)
    return model.predict(dmat)[0]

# --- 2. CONFIGURATION ---
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css",
    "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "ChurnGuard AI"

# --- 3. STYLING ---
glass_card = {
    'backgroundColor': 'rgba(255, 255, 255, 0.05)',
    'backdropFilter': 'blur(10px)',
    'boxShadow': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
    'border': '1px solid rgba(255, 255, 255, 0.1)',
    'borderRadius': '15px',
    'color': 'white'
}

themes = {
    'dark': {
        'bg_gradient': 'linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)',
        'text': '#ffffff', 'sub_text': '#d1d1d1', 'accent': '#00d2ff',
        'card_style': glass_card, 'btn_icon': 'bi bi-sun-fill',
        'gauge_bg': "rgba(255,255,255,0.1)", 'gauge_border': "rgba(255,255,255,0.3)",
        'btn_style': {'backgroundColor': 'rgba(255,255,255,0.1)', 'color': '#ffc107', 'border': '1px solid rgba(255,255,255,0.2)', 'borderRadius': '50%', 'width': '50px', 'height': '50px', 'fontSize': '1.5rem', 'boxShadow': '0 0 15px rgba(255, 193, 7, 0.4)'}
    },
    'light': {
        'bg_gradient': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        'text': '#2c3e50', 'sub_text': '#586776', 'accent': '#3a7bd5',
        'card_style': {'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'boxShadow': '0 8px 32px 0 rgba(31, 38, 135, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.4)', 'borderRadius': '15px'},
        'btn_icon': 'bi bi-moon-stars-fill',
        'gauge_bg': "rgba(0,0,0,0.1)", 'gauge_border': "rgba(0,0,0,0.2)",
        'btn_style': {'backgroundColor': 'rgba(255,255,255,0.8)', 'color': '#3a7bd5', 'border': '1px solid rgba(0,0,0,0.1)', 'borderRadius': '50%', 'width': '50px', 'height': '50px', 'fontSize': '1.2rem', 'boxShadow': '0 0 15px rgba(58, 123, 213, 0.4)'}
    }
}

# --- 4. LAYOUT ---
app.layout = html.Div(id='main-page', style={'minHeight': '100vh', 'fontFamily': 'Poppins, sans-serif', 'transition': 'background 0.5s'}, children=[
    
    dbc.Container([
        # HEADER
        dbc.Row([
            dbc.Col([
                html.H1([html.I(className="bi bi-robot me-2"), "ChurnGuard AI"], 
                        id='app-title', className="text-start text-md-center", style={'fontWeight': '600', 'letterSpacing': '1px'}),
                html.P("Real-time Customer Retention Analytics", 
                       id='app-subtitle', className="text-start text-md-center mb-0", style={'fontSize': '0.9rem'})
            ], width=8, md=8, className="d-flex flex-column justify-content-center"),
            
            dbc.Col([
                dbc.Button(html.I(id='theme-icon', className="bi bi-sun-fill"), id='theme-toggle', n_clicks=0, className="d-flex align-items-center justify-content-center")
            ], width=4, md=4, className="d-flex justify-content-end align-items-center")
        ], className="mb-5 align-items-center"),

        dbc.Row([
            # LEFT COLUMN: INPUTS
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.H4([html.I(className="bi bi-sliders me-2"), "Profile"], id='input-title', className="card-title"), width=6),
                            dbc.Col([
                                dbc.Button(html.I(className="bi bi-person-plus-fill"), id="open-add-modal", size="sm", color="success", className="me-1"),
                                dbc.Button(html.I(className="bi bi-arrow-clockwise"), id="btn-retrain", size="sm", color="warning", title="Retrain AI on new data") 
                            ], width=6, className="d-flex justify-content-end")
                        ], className="mb-4 align-items-center"),
                        html.Div(id='retrain-msg', className="text-center small fw-bold mb-3"),
                        
                        html.Label("Search Customer ID", className="small fw-bold"),
                        dcc.Dropdown(id='search-dropdown', options=[], placeholder="Type to search...", className="mb-3 text-dark"),
                        html.Div(id='search-msg', className="text-center small mb-3", style={'minHeight': '20px'}),

                        html.Label("Tenure (Months)", id='lbl-tenure', className="fw-bold mt-2"),
dcc.Slider(
    id='tenure-slider', 
    min=0, 
    max=72, 
    step=1, 
    value=12, 
    # This 'marks' property fixes the visual clutter
    marks={i: f'{i}m' for i in range(0, 73, 12)}, 
    tooltip={"placement": "bottom", "always_visible": True}
),

html.Label("Monthly Charges ($)", id='lbl-charges', className="fw-bold mt-4"),
dcc.Slider(
    id='charges-slider', 
    min=20, 
    max=120, 
    step=1, 
    value=70, 
    # Show a label every $20 instead of every $1
    marks={i: f'${i}' for i in range(20, 121, 20)}, 
    tooltip={"placement": "bottom", "always_visible": True}
),
                        
                        html.Label("Contract Type", id='lbl-contract', className="fw-bold mt-4 mb-2"),
                        dcc.Dropdown(id='contract-dropdown', options=[{'label': 'Month-to-Month', 'value': 'Month'}, {'label': 'One Year', 'value': 'OneYear'}, {'label': 'Two Year', 'value': 'TwoYear'}], value='Month', clearable=False, style={'color': '#333'}),
                        
                        # SAVE SNAPSHOT BUTTON
                        html.Hr(className="mt-4"),
                        dbc.Button([html.I(className="bi bi-camera-fill me-2"), "Save Snapshot"], id="btn-save-snapshot", color="info", className="w-100", disabled=True),
                        html.Div(id="save-msg", className="text-center small mt-2")

                    ])
                ], id='input-card', style=glass_card)
            ], width=12, lg=4, className="mb-4", style={'zIndex': 100, 'position': 'relative'}),

            # RIGHT COLUMN: TABS (ANALYTICS & HISTORY)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Tabs([
                            # TAB 1: DASHBOARD
                            dbc.Tab(label="Analytics Dashboard", tab_id="tab-dashboard", label_style={"color": "#00d2ff"}, children=[
                                html.Div(className="mt-4", children=[
                                    dbc.Row([
                                        dbc.Col([
                                            html.H4([html.I(className="bi bi-speedometer2 me-2"), "Risk Score"], id='res-title-1', className="text-center"),
                                            dcc.Graph(id='churn-gauge', config={'displayModeBar': False}, style={'height': '200px'}),
                                            html.Div(id='prediction-badge', className="text-center mt-2")
                                        ], width=12, md=5),

                                        dbc.Col([
                                            html.H4([html.I(className="bi bi-graph-up-arrow me-2"), "Projection"], id='res-title-2', className="text-center"),
                                            html.P("Projected risk over next 12 months.", id='txt-hint-proj', className="text-center small"),
                                            dcc.Graph(id='trend-graph', config={'displayModeBar': False}, style={'height': '150px'})
                                        ], width=12, md=5, className="d-flex flex-column justify-content-center mt-5 mt-md-0")
                                    ], className="g-0", justify="between"),

                                    html.Div([dbc.Alert(id='recommendation-alert', color='info', className="d-flex align-items-center mt-4", style={'backgroundColor': 'rgba(0, 210, 255, 0.15)', 'border': '1px solid rgba(0, 210, 255, 0.3)', 'color': 'white'})]),
                                    html.Hr(className="my-4"),
                                    dbc.Row([dbc.Col([html.H5([html.I(className="bi bi-list-check me-2"), "Top Risk Drivers"], id='res-title-3', className="mb-3"), dcc.Graph(id='feature-importance-graph', config={'displayModeBar': False}, style={'height': '200px'})], width=12)])
                                ])
                            ]),
                            
                            # TAB 2: HISTORY
                            dbc.Tab(label="Customer History", tab_id="tab-history", label_style={"color": "#ffc107"}, children=[
                                html.Div(className="mt-4", children=[
                                    html.H5("Risk Score History", className="mb-3"),
                                    dcc.Graph(id='history-graph', config={'displayModeBar': False}, style={'height': '300px'}),
                                    html.Hr(),
                                    html.H5("Recent Snapshots", className="mb-3"),
                                    html.Div(id='history-table-container')
                                ])
                            ])
                        ], id="tabs", active_tab="tab-dashboard")
                    ])
                ], id='result-card', style=glass_card)
            ], width=12, lg=8)
        ]),

        # MODAL
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Add New Customer"), close_button=True),
            dbc.ModalBody([
                html.Label("Tenure"), dbc.Input(id="new-tenure", type="number", value=1, className="mb-3"),
                html.Label("Monthly Charges"), dbc.Input(id="new-charges", type="number", value=50, className="mb-3"),
                html.Label("Contract"), dcc.Dropdown(id='new-contract', options=[{'label': 'Month', 'value': 'Month-to-month'}, {'label': '1 Year', 'value': 'One year'}, {'label': '2 Year', 'value': 'Two year'}], value='Month-to-month', clearable=False, className="mb-3 text-dark"),
                html.Div(id="add-user-msg", className="text-danger small")
            ]),
            dbc.ModalFooter([dbc.Button("Save", id="save-new-user", color="primary")])
        ], id="add-modal", is_open=False)

    ], fluid=True, style={'maxWidth': '1400px'})
])

# --- 5. CALLBACKS ---

# --- A. SEARCH + ENABLE SNAPSHOT BTN ---

# --- NEW: DYNAMIC SEARCH (Prevents Memory Crash) ---
@app.callback(
    Output("search-dropdown", "options"),
    Input("search-dropdown", "search_value")
)
def update_search_options(search_value):
    if not search_value:
        # Prevent DB query if user hasn't typed anything
        return dash.no_update
    
    try:
        # Only connect when user types
        conn = sqlite3.connect('churn.db')
        
        # Search for IDs starting with the typed text (LIMIT 10 for speed)
        query = f"SELECT customerID FROM customers WHERE customerID LIKE '{search_value}%' LIMIT 10"
        df_results = pd.read_sql(query, conn)
        conn.close()
        
        # Format for Dropdown
        return [{'label': i, 'value': i} for i in df_results['customerID']]
    except:
        return []
    
@app.callback(
    [Output('tenure-slider', 'value'), Output('charges-slider', 'value'), Output('contract-dropdown', 'value'), Output('search-msg', 'children'), Output('search-msg', 'style'), Output('btn-save-snapshot', 'disabled')],
    [Input('search-dropdown', 'value')] 
)
def search_customer_dropdown(customer_id):
    if not customer_id:
        return dash.no_update, dash.no_update, dash.no_update, "", {}, True
    try:
        conn = sqlite3.connect('churn.db')
        query = f"SELECT tenure, MonthlyCharges, Contract FROM customers WHERE customerID = '{customer_id}'"
        df_result = pd.read_sql(query, conn)
        conn.close()
        if df_result.empty:
            return dash.no_update, dash.no_update, dash.no_update, "❌ Not found.", {'color': 'red'}, True
        row = df_result.iloc[0]
        contract_map = {'Month-to-month': 'Month', 'One year': 'OneYear', 'Two year': 'TwoYear'}
        mapped_contract = contract_map.get(row['Contract'], 'Month')
        return row['tenure'], row['MonthlyCharges'], mapped_contract, f"✅ Loaded: {customer_id}", {'color': '#2ECC40'}, False
    except:
        return dash.no_update, dash.no_update, dash.no_update, "DB Error", {'color': 'red'}, True

# --- B. SAVE SNAPSHOT LOGIC ---
@app.callback(
    Output('save-msg', 'children'),
    Input('btn-save-snapshot', 'n_clicks'),
    [State('search-dropdown', 'value'), State('tenure-slider', 'value'), State('charges-slider', 'value'), State('contract-dropdown', 'value')]
)
def save_snapshot(n_clicks, customer_id, tenure, charges, contract):
    if not n_clicks or not customer_id: return ""
    
    # Calculate Risk Score for this snapshot
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = charges
    input_data['TotalCharges'] = tenure * charges
    if contract == 'OneYear': input_data['Contract_One year'] = 1
    elif contract == 'TwoYear': input_data['Contract_Two year'] = 1
    elif contract == 'Month': input_data['Contract_Month-to-month'] = 1
    
    try:
        risk_score = safe_predict(input_data)
        
        conn = sqlite3.connect('churn.db')
        cursor = conn.cursor()
        # Create table if not exists (Safety check)
        cursor.execute('''CREATE TABLE IF NOT EXISTS risk_history (id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, risk_score REAL, tenure INTEGER, monthly_charges REAL, contract TEXT)''')
        
        cursor.execute("INSERT INTO risk_history (customer_id, risk_score, tenure, monthly_charges, contract) VALUES (?, ?, ?, ?, ?)", 
                       (customer_id, float(risk_score), tenure, charges, contract))
        conn.commit()
        conn.close()
        return f"✅ Snapshot saved! Risk: {risk_score*100:.1f}%"
    except Exception as e:
        return f"❌ Error saving: {str(e)}"

# --- C. UPDATE HISTORY TAB ---
@app.callback(
    [Output('history-graph', 'figure'), Output('history-table-container', 'children')],
    [Input('tabs', 'active_tab'), 
     Input('search-dropdown', 'value'), 
     Input('save-msg', 'children')] # <--- CHANGED THIS: Listens to the confirmation message, not the button click
)
def update_history_tab(active_tab, customer_id, save_msg_trigger):
    # 1. Basic Checks
    if active_tab != "tab-history" or not customer_id:
        return go.Figure(), "Select a customer to view history."
    
    try:
        # 2. Use Context Manager for Safety
        with sqlite3.connect('churn.db', timeout=10) as conn:
            
            # Check table existence
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_history'")
            if not cursor.fetchone():
                return go.Figure(), "No history yet. Click 'Save Snapshot' to start tracking."
            
            # Read Data
            df_hist = pd.read_sql(f"SELECT prediction_date, risk_score, tenure, monthly_charges FROM risk_history WHERE customer_id = '{customer_id}' ORDER BY prediction_date ASC", conn)
            
            if df_hist.empty:
                return go.Figure(), "No history found for this customer. Save a snapshot first."
            
            # 3. Create Graph
            fig = px.line(df_hist, x='prediction_date', y='risk_score', markers=True, title=f"Risk Trend: {customer_id}")
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0, 1]))
            
            # 4. Create Table
            table = dbc.Table.from_dataframe(df_hist.tail(5), striped=True, bordered=True, hover=True, className="text-white")
            
            return fig, table

    except Exception as e:
        print(f"❌ HISTORY ERROR: {e}") 
        return go.Figure(), "Error loading history."

# --- D. ADD USER MODAL ---
@app.callback(
    [Output("add-modal", "is_open"), 
     Output("search-dropdown", "options", allow_duplicate=True), # <--- ADD allow_duplicate=True
     Output("search-dropdown", "value"), 
     Output("add-user-msg", "children")],
    [Input("open-add-modal", "n_clicks"), Input("save-new-user", "n_clicks")],
    [State("add-modal", "is_open"), State("new-tenure", "value"), State("new-charges", "value"), State("new-contract", "value"), State("search-dropdown", "options")],
    prevent_initial_call=True # <--- ADD THIS at the end
)

def toggle_modal(open_clicks, save_clicks, is_open, tenure, charges, contract, current_options):
    trigger = ctx.triggered_id
    if trigger == "open-add-modal": return True, dash.no_update, dash.no_update, ""
    if trigger == "save-new-user":
        if tenure is None or charges is None: return True, dash.no_update, dash.no_update, "❌ Fill all fields."
        new_id = f"NEW-{random.randint(1000, 9999)}"
        try:
            conn = sqlite3.connect('churn.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO customers (customerID, tenure, MonthlyCharges, TotalCharges, Contract, Churn) VALUES (?, ?, ?, ?, ?, ?)", (new_id, tenure, charges, tenure*charges, contract, 'No'))
            conn.commit()
            conn.close()
            return False, [new_id] + current_options, new_id, "" 
        except Exception as e: return True, dash.no_update, dash.no_update, f"Error: {str(e)}"
    return is_open, dash.no_update, dash.no_update, ""

# --- E. ANALYTICS & THEME ---
@app.callback(
    [Output('main-page', 'style'), Output('app-title', 'style'), Output('app-subtitle', 'style'), Output('theme-toggle', 'style'), Output('theme-icon', 'className'), Output('input-card', 'style'), Output('result-card', 'style'), Output('input-title', 'style'), Output('res-title-1', 'style'), Output('res-title-2', 'style'), Output('res-title-3', 'style'), Output('lbl-tenure', 'style'), Output('lbl-charges', 'style'), Output('lbl-contract', 'style')],
    [Input('theme-toggle', 'n_clicks')]
)
def toggle_theme(n_clicks):
    is_dark = (n_clicks % 2 == 0)
    t = themes['dark'] if is_dark else themes['light']
    page_style = {'background': t['bg_gradient'], 'minHeight': '100vh', 'fontFamily': 'Poppins, sans-serif', 'padding': '20px', 'transition': '0.5s'}
    title_style = {'color': t['text']}
    sub_style = {'color': t['sub_text']}
    return (page_style, title_style, sub_style, t['btn_style'], t['btn_icon'], t['card_style'], t['card_style'], title_style, title_style, title_style, title_style, title_style, title_style, title_style)

@app.callback(
    [Output('churn-gauge', 'figure'), Output('trend-graph', 'figure'), Output('prediction-badge', 'children'), Output('feature-importance-graph', 'figure'), Output('recommendation-alert', 'children')],
    [Input('tenure-slider', 'value'), Input('charges-slider', 'value'), Input('contract-dropdown', 'value'), Input('theme-toggle', 'n_clicks')]
)
def update_dashboard(tenure, monthly_charges, contract_type, n_clicks):
    is_dark = (n_clicks % 2 == 0)
    t = themes['dark'] if is_dark else themes['light']
    
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = tenure * monthly_charges
    
    if contract_type == 'OneYear': input_data['Contract_One year'] = 1
    elif contract_type == 'TwoYear': input_data['Contract_Two year'] = 1
    elif contract_type == 'Month': input_data['Contract_Month-to-month'] = 1

    try: prob = safe_predict(input_data)
    except: prob = 0.5 

    sim_price_drop = 0
    if monthly_charges > 30: 
        sim_data = input_data.copy()
        sim_data['MonthlyCharges'] -= 20
        sim_data['TotalCharges'] = tenure * sim_data['MonthlyCharges']
        sim_price_drop = prob - safe_predict(sim_data)

    sim_contract_drop = 0
    if contract_type == 'Month': 
        sim_data = input_data.copy()
        for c in sim_data.columns: 
            if 'Contract_' in c: sim_data[c] = 0
        if 'Contract_One year' in sim_data.columns: sim_data['Contract_One year'] = 1
        sim_contract_drop = prob - safe_predict(sim_data)

    if prob < 0.3: rec_text = [html.I(className="bi bi-check-circle-fill me-2"), "Safe. No action needed."]
    elif sim_contract_drop > sim_price_drop and sim_contract_drop > 0.05: rec_text = [html.I(className="bi bi-file-earmark-text-fill me-2"), f"Recommendation: Switch to 1-Year Contract (Risk -{(sim_contract_drop*100):.1f}%)"]
    elif sim_price_drop > 0.05: rec_text = [html.I(className="bi bi-currency-dollar me-2"), f"Recommendation: Lower charges by $20 (Risk -{(sim_price_drop*100):.1f}%)"]
    else: rec_text = [html.I(className="bi bi-exclamation-triangle-fill me-2"), "High Risk: Personal retention call advised."]

    bar_color = "#ff4b1f" if prob > 0.5 else "#00f2c3"
    status_text = "HIGH RISK" if prob > 0.5 else "SAFE"
    badge_color = "danger" if prob > 0.5 else "success"

    fig_gauge = go.Figure(go.Indicator(mode = "gauge+number", value = prob * 100, number = {'suffix': "%", 'font': {'color': t['text'], 'size': 40}}, gauge = {'axis': {'range': [0, 100], 'tickcolor': t['text']}, 'bar': {'color': bar_color}, 'bgcolor': t['gauge_bg'], 'bordercolor': t['gauge_border']}))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'family': "Poppins"}, margin={'t': 20, 'b': 0, 'l': 20, 'r': 20})

    future_tenure = [tenure + i for i in range(0, 13)]
    risk_trend = [max(0, prob - ((ft - tenure) * 0.025)) * 100 for ft in future_tenure]
    fig_trend = go.Figure(go.Scatter(x=list(range(0, 13)), y=risk_trend, mode='lines', fill='tozeroy', line=dict(color=t['accent'], width=3, shape='spline')))
    fig_trend.update_layout(template='plotly_dark' if is_dark else 'plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin={'t': 10, 'b': 20, 'l': 30, 'r': 10}, height=150, xaxis={'showgrid': False}, yaxis={'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.1)'})

    try:
        imp_df = pd.DataFrame(list(model.get_score(importance_type='gain').items()), columns=['Feature', 'Score']).sort_values(by='Score', ascending=True).tail(5)
        fig_imp = go.Figure(go.Bar(x=imp_df['Score'], y=imp_df['Feature'], orientation='h', marker_color=t['accent'], opacity=0.8))
        fig_imp.update_layout(template='plotly_dark' if is_dark else 'plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin={'t': 0, 'b': 0, 'l': 150, 'r': 20}, height=200, xaxis={'visible': False}, yaxis={'tickfont': {'color': t['text']}})
    except: fig_imp = go.Figure()

    badge = dbc.Badge([html.I(className="bi bi-activity me-2"), status_text], color=badge_color, className="p-2", style={'fontSize': '1rem'})
    return fig_gauge, fig_trend, badge, fig_imp, rec_text

@app.callback(Output('retrain-msg', 'children'), Input('btn-retrain', 'n_clicks'), prevent_initial_call=True)
def retrain_callback(n):
    msg = retrain.run_retraining()
    load_model_resources()
    return msg

if __name__ == '__main__':
    app.run(debug=True)