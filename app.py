import sqlite3
import json
import dash
import random
import string
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import xgboost as xgb
import pandas as pd
import plotly.graph_objects as go

# --- 1. SETUP & LOAD RESOURCES ---
model = xgb.Booster()
model.load_model('model.json')

# Load Feature Names
try:
    with open('features.json', 'r') as f:
        feature_names = json.load(f)
except FileNotFoundError:
    feature_names = []

# Fetch All IDs for Autocomplete
def get_all_ids():
    try:
        conn = sqlite3.connect('churn.db')
        # Get just the IDs list
        ids = pd.read_sql("SELECT customerID FROM customers", conn)['customerID'].tolist()
        conn.close()
        return ids
    except:
        return []

all_customer_ids = get_all_ids()

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
       # --- HEADER (Aligned to the 'Gap' on Desktop, Side-by-Side on Mobile) ---
        dbc.Row([
            # 1. Title Column (Span 8 cols)
            # Desktop: Centering text here puts it exactly at the 33% mark (the gap).
            dbc.Col([
                html.H1([html.I(className="bi bi-robot me-2"), "ChurnGuard AI"], 
                        id='app-title', className="text-start text-md-center", style={'fontWeight': '600', 'letterSpacing': '1px'}),
                html.P("Real-time Customer Retention Analytics", 
                       id='app-subtitle', className="text-start text-md-center mb-0", style={'fontSize': '0.9rem'})
            ], width=8, md=8, className="d-flex flex-column justify-content-center"),
            
            # 2. Button Column (Span 4 cols)
            dbc.Col([
                dbc.Button(
                    html.I(id='theme-icon', className="bi bi-sun-fill"), 
                    id='theme-toggle', 
                    n_clicks=0, 
                    className="d-flex align-items-center justify-content-center"
                )
            ], width=4, md=4, className="d-flex justify-content-end align-items-center")
        ], className="mb-5 align-items-center"),

        # --- DASHBOARD CONTENT ---
        dbc.Row([
            # --- LEFT: INPUTS ---
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        # TITLE + ADD BUTTON ROW
                        dbc.Row([
                            dbc.Col(html.H4([html.I(className="bi bi-sliders me-2"), "Profile"], id='input-title', className="card-title"), width=8),
                            dbc.Col(dbc.Button([html.I(className="bi bi-person-plus-fill me-1"), "New"], id="open-add-modal", size="sm", color="success", className="w-100"), width=4)
                        ], className="mb-4 align-items-center"),
                        
                        # --- SMART SEARCH DROPDOWN ---
                        html.Label("Search Customer ID", className="small fw-bold"),
                        dcc.Dropdown(
                            id='search-dropdown',
                            options=all_customer_ids, # Loads all IDs
                            placeholder="Type to search (e.g. 7590...)",
                            className="mb-3 text-dark"
                        ),
                        html.Div(id='search-msg', className="text-center small mb-3", style={'minHeight': '20px'}),
                        # -----------------------------

                        html.Label("Tenure (Months)", id='lbl-tenure', className="fw-bold mt-2"),
                        dcc.Slider(id='tenure-slider', min=0, max=72, step=1, value=12, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.Label("Monthly Charges ($)", id='lbl-charges', className="fw-bold mt-4"),
                        dcc.Slider(id='charges-slider', min=20, max=120, step=1, value=70, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.Label("Contract Type", id='lbl-contract', className="fw-bold mt-4 mb-2"),
                        dcc.Dropdown(id='contract-dropdown', options=[{'label': 'Month-to-Month', 'value': 'Month'}, {'label': 'One Year', 'value': 'OneYear'}, {'label': 'Two Year', 'value': 'TwoYear'}], value='Month', clearable=False, style={'color': '#333'}),
                        
                        html.Div(className="mt-4", children=[html.Small("Adjust parameters to see real-time risk impact.", id='txt-hint-input')])
                    ])
                ], id='input-card', style=glass_card)
            ], width=12, lg=4, className="mb-4", style={'zIndex': 100, 'position': 'relative'}),

            # --- RIGHT: ANALYTICS ---
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
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

                        html.Div([dbc.Alert(id='recommendation-alert', color='info', className="d-flex align-items-center", style={'backgroundColor': 'rgba(0, 210, 255, 0.15)', 'border': '1px solid rgba(0, 210, 255, 0.3)', 'color': 'white'})], className="mt-4"),
                        html.Hr(className="my-4"),
                        dbc.Row([dbc.Col([html.H5([html.I(className="bi bi-list-check me-2"), "Top Risk Drivers (Global)"], id='res-title-3', className="mb-3"), dcc.Graph(id='feature-importance-graph', config={'displayModeBar': False}, style={'height': '200px'})], width=12)])
                    ])
                ], id='result-card', style=glass_card)
            ], width=12, lg=8)
        ]),

        # --- ADD USER MODAL (POPUP) ---
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Add New Customer"), close_button=True),
            dbc.ModalBody([
                html.Label("Tenure (Months)"),
                dbc.Input(id="new-tenure", type="number", value=1, min=0, max=72, className="mb-3"),
                html.Label("Monthly Charges ($)"),
                dbc.Input(id="new-charges", type="number", value=50, min=0, max=200, className="mb-3"),
                html.Label("Contract Type"),
                dcc.Dropdown(id='new-contract', options=[
                    {'label': 'Month-to-Month', 'value': 'Month-to-month'},
                    {'label': 'One Year', 'value': 'One year'},
                    {'label': 'Two Year', 'value': 'Two year'}
                ], value='Month-to-month', clearable=False, className="mb-3 text-dark"),
                html.Div(id="add-user-msg", className="text-danger small")
            ]),
            dbc.ModalFooter([
                dbc.Button("Save to Database", id="save-new-user", color="primary")
            ])
        ], id="add-modal", is_open=False)

    ], fluid=True, style={'maxWidth': '1400px'})
])

# --- 5. CALLBACKS ---

# --- A. SEARCH CUSTOMER (AUTOCOMPLETE) ---
@app.callback(
    [Output('tenure-slider', 'value'),
     Output('charges-slider', 'value'),
     Output('contract-dropdown', 'value'),
     Output('search-msg', 'children'),
     Output('search-msg', 'style')],
    [Input('search-dropdown', 'value')] # Triggered when user selects from dropdown
)
def search_customer_dropdown(customer_id):
    if not customer_id:
        return dash.no_update, dash.no_update, dash.no_update, "", {}

    try:
        conn = sqlite3.connect('churn.db')
        query = f"SELECT tenure, MonthlyCharges, Contract FROM customers WHERE customerID = '{customer_id}'"
        df_result = pd.read_sql(query, conn)
        conn.close()

        if df_result.empty:
            return dash.no_update, dash.no_update, dash.no_update, "❌ ID not found.", {'color': '#FF4136'}

        row = df_result.iloc[0]
        contract_map = {'Month-to-month': 'Month', 'One year': 'OneYear', 'Two year': 'TwoYear'}
        mapped_contract = contract_map.get(row['Contract'], 'Month')

        return row['tenure'], row['MonthlyCharges'], mapped_contract, f"✅ Loaded: {customer_id}", {'color': '#2ECC40'}
    except:
        return dash.no_update, dash.no_update, dash.no_update, "DB Error", {'color': 'red'}


# --- B. ADD USER MODAL LOGIC ---
@app.callback(
    [Output("add-modal", "is_open"),
     Output("search-dropdown", "options"), # Update dropdown list after adding!
     Output("search-dropdown", "value"),   # Auto-select the new user
     Output("add-user-msg", "children")],
    [Input("open-add-modal", "n_clicks"),
     Input("save-new-user", "n_clicks")],
    [State("add-modal", "is_open"),
     State("new-tenure", "value"),
     State("new-charges", "value"),
     State("new-contract", "value"),
     State("search-dropdown", "options")]
)
def toggle_modal(open_clicks, save_clicks, is_open, tenure, charges, contract, current_options):
    trigger = ctx.triggered_id

    # 1. Open Modal
    if trigger == "open-add-modal":
        return True, dash.no_update, dash.no_update, ""

    # 2. Save Data
    if trigger == "save-new-user":
        if tenure is None or charges is None:
            return True, dash.no_update, dash.no_update, "❌ Please fill all fields."
        
        # Generate Fake ID (e.g., "NEW-8239")
        new_id = f"NEW-{random.randint(1000, 9999)}"
        
        try:
            conn = sqlite3.connect('churn.db')
            cursor = conn.cursor()
            # INSERT SQL command
            cursor.execute(
                "INSERT INTO customers (customerID, tenure, MonthlyCharges, TotalCharges, Contract, Churn) VALUES (?, ?, ?, ?, ?, ?)",
                (new_id, tenure, charges, tenure*charges, contract, 'No')
            )
            conn.commit()
            conn.close()
            
            # Add new ID to dropdown options
            new_options = [new_id] + current_options
            
            return False, new_options, new_id, "" # Close modal, update list, select new user
            
        except Exception as e:
            return True, dash.no_update, dash.no_update, f"Error: {str(e)}"

    return is_open, dash.no_update, dash.no_update, ""


# --- C. EXISTING VISUALIZATION LOGIC ---
@app.callback(
    [Output('main-page', 'style'), Output('app-title', 'style'), Output('app-subtitle', 'style'), Output('theme-toggle', 'style'), Output('theme-icon', 'className'), Output('input-card', 'style'), Output('result-card', 'style'), Output('input-title', 'style'), Output('res-title-1', 'style'), Output('res-title-2', 'style'), Output('res-title-3', 'style'), Output('lbl-tenure', 'style'), Output('lbl-charges', 'style'), Output('lbl-contract', 'style'), Output('txt-hint-input', 'style'), Output('txt-hint-proj', 'style')],
    [Input('theme-toggle', 'n_clicks')]
)
def toggle_theme(n_clicks):
    is_dark = (n_clicks % 2 == 0)
    t = themes['dark'] if is_dark else themes['light']
    page_style = {'background': t['bg_gradient'], 'minHeight': '100vh', 'fontFamily': 'Poppins, sans-serif', 'padding': '20px', 'transition': '0.5s'}
    title_style = {'color': t['text']}
    sub_style = {'color': t['sub_text']}
    return (page_style, title_style, sub_style, t['btn_style'], t['btn_icon'], t['card_style'], t['card_style'], title_style, title_style, title_style, title_style, title_style, title_style, title_style, sub_style, sub_style)

@app.callback(
    [Output('churn-gauge', 'figure'), Output('trend-graph', 'figure'), Output('prediction-badge', 'children'), Output('feature-importance-graph', 'figure'), Output('recommendation-alert', 'children'), Output('recommendation-alert', 'style')],
    [Input('tenure-slider', 'value'), Input('charges-slider', 'value'), Input('contract-dropdown', 'value'), Input('theme-toggle', 'n_clicks')]
)
def update_analytics(tenure, monthly_charges, contract_type, n_clicks):
    is_dark = (n_clicks % 2 == 0)
    t = themes['dark'] if is_dark else themes['light']
    
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = tenure * monthly_charges
    if contract_type == 'OneYear': input_data['Contract_One year'] = 1
    elif contract_type == 'TwoYear': input_data['Contract_Two year'] = 1
    
    dmatrix = xgb.DMatrix(input_data)
    prob = model.predict(dmatrix)[0]
    
    # Simulation
    sim_price_drop = 0
    if monthly_charges > 30: 
        sim_data_price = input_data.copy()
        sim_data_price['MonthlyCharges'] = monthly_charges - 20
        sim_data_price['TotalCharges'] = tenure * (monthly_charges - 20)
        prob_price = model.predict(xgb.DMatrix(sim_data_price))[0]
        sim_price_drop = prob - prob_price 

    sim_contract_drop = 0
    if contract_type == 'Month': 
        sim_data_contract = input_data.copy()
        sim_data_contract['Contract_One year'] = 1 
        sim_data_contract['Contract_Two year'] = 0
        prob_contract = model.predict(xgb.DMatrix(sim_data_contract))[0]
        sim_contract_drop = prob - prob_contract

    if prob < 0.3:
        rec_text = [html.I(className="bi bi-check-circle-fill me-2"), "Customer is safe. No immediate action needed."]
        rec_style = {'backgroundColor': 'rgba(46, 204, 64, 0.2)', 'border': '1px solid #2ECC40', 'color': t['text']}
    elif sim_contract_drop > sim_price_drop and sim_contract_drop > 0.05:
        rec_text = [html.I(className="bi bi-file-earmark-text-fill me-2"), html.Span(f"💡 AI Recommendation: Switching to a 1-Year Contract could lower risk by {(sim_contract_drop*100):.1f}%.")]
        rec_style = {'backgroundColor': 'rgba(255, 220, 0, 0.2)', 'border': '1px solid #FFDC00', 'color': t['text']}
    elif sim_price_drop > 0.05:
        rec_text = [html.I(className="bi bi-currency-dollar me-2"), html.Span(f"💡 AI Recommendation: Lowering monthly charges by $20 could lower risk by {(sim_price_drop*100):.1f}%.")]
        rec_style = {'backgroundColor': 'rgba(0, 210, 255, 0.2)', 'border': '1px solid #00d2ff', 'color': t['text']}
    else:
        rec_text = [html.I(className="bi bi-exclamation-triangle-fill me-2"), "High Risk: Consider a personalized retention call."]
        rec_style = {'backgroundColor': 'rgba(255, 65, 54, 0.2)', 'border': '1px solid #FF4136', 'color': t['text']}

    # Visuals
    bar_color = "#ff4b1f" if prob > 0.5 else "#00f2c3"
    status_text = "HIGH RISK" if prob > 0.5 else "SAFE"
    badge_color = "danger" if prob > 0.5 else "success"

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = prob * 100, number = {'suffix': "%", 'font': {'color': t['text'], 'size': 40}},
        gauge = {'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': t['text']}, 'bar': {'color': bar_color, 'thickness': 0.75}, 'bgcolor': t['gauge_bg'], 'bordercolor': t['gauge_border'], 'borderwidth': 2, 'threshold': {'line': {'color': t['text'], 'width': 4}, 'thickness': 0.75, 'value': prob*100}}
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'family': "Poppins"}, margin={'t': 20, 'b': 0, 'l': 20, 'r': 20})

    future_tenure = [tenure + i for i in range(0, 13)]
    risk_trend = [max(0, prob - ((ft - tenure) * 0.025)) * 100 for ft in future_tenure]
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=list(range(0, 13)), y=risk_trend, mode='lines', fill='tozeroy', line=dict(color=t['accent'], width=3, shape='spline')))
    fig_trend.update_layout(template='plotly_dark' if is_dark else 'plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin={'t': 10, 'b': 20, 'l': 30, 'r': 10}, height=150, xaxis={'title': '+ Months', 'showgrid': False}, yaxis={'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.1)'})

    importance_map = model.get_score(importance_type='gain')
    imp_df = pd.DataFrame(list(importance_map.items()), columns=['Feature', 'Score']).sort_values(by='Score', ascending=True).tail(5) 
    fig_imp = go.Figure(go.Bar(x=imp_df['Score'], y=imp_df['Feature'], orientation='h', marker_color=t['accent'], opacity=0.8))
    fig_imp.update_layout(template='plotly_dark' if is_dark else 'plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin={'t': 0, 'b': 0, 'l': 150, 'r': 20}, height=200, xaxis={'showgrid': False, 'visible': False}, yaxis={'tickfont': {'color': t['text'], 'family': 'Poppins'}})

    badge = dbc.Badge([html.I(className="bi bi-activity me-2"), status_text], color=badge_color, className="p-2", style={'fontSize': '1rem'})
    return fig_gauge, fig_trend, badge, fig_imp, rec_text, rec_style

if __name__ == '__main__':
    app.run(debug=True)