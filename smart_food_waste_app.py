#!/usr/bin/env python
# coding: utf-8

# smart_food_waste_app.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from dash import Dash, dcc, html, Output, Input
import plotly.express as px

# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
df = pd.read_csv("C:/Users/dell/Downloads/global_food_wastage_dataset.csv")

# Automatic column mapping
column_mapping = {}
for col in df.columns:
    col_lower = col.strip().lower().replace(" ", "_")
    if "country" in col_lower:
        column_mapping[col] = "country"
    elif "food" in col_lower and "category" in col_lower:
        column_mapping[col] = "food_category"
    elif "year" in col_lower:
        column_mapping[col] = "year"
    elif "total" in col_lower and "waste" in col_lower:
        column_mapping[col] = "total_waste_tons"
    elif "economic" in col_lower and "loss" in col_lower:
        column_mapping[col] = "economic_loss_million_usd"
    elif "avg" in col_lower and "waste" in col_lower:
        column_mapping[col] = "avg_waste_per_capita"
    elif "population" in col_lower:
        column_mapping[col] = "population_millions"

df.rename(columns=column_mapping, inplace=True)

# Clean data
df["year"] = pd.to_numeric(df["year"], errors='coerce')
df.dropna(subset=["year", "total_waste_tons", "avg_waste_per_capita", "population_millions"], inplace=True)
df = df[df["population_millions"] > 0]
df["year"] = df["year"].astype(int)
df["waste_per_million_pop"] = df["total_waste_tons"] / df["population_millions"]

# -----------------------------
# 2. Train ML model for prediction
# -----------------------------
X = df[["year","population_millions","avg_waste_per_capita"]]
y = df["total_waste_tons"]
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# 3. Dash App Setup
# -----------------------------
app = Dash(__name__)
app.title = "Smart Food Waste Reduction System"

# App Layout
app.layout = html.Div([
    html.H1("🌍 AI-Based Smart Food Waste Reduction Dashboard", style={'textAlign':'center'}),
    
    # Global Trends Section
    html.H2("Global Food Waste Trends"),
    html.Label("Select Country(s):"),
    dcc.Dropdown(
        options=[{'label': c, 'value': c} for c in df["country"].unique()],
        value=["India"],
        multi=True,
        id="country-dropdown"
    ),
    html.Label("Select Year Range:"),
    dcc.RangeSlider(
        min=int(df["year"].min()),
        max=int(df["year"].max()),
        step=1,
        value=[2010, 2020],
        marks={year: str(year) for year in range(int(df["year"].min()), int(df["year"].max())+1, 2)},
        id="year-slider"
    ),
    dcc.Graph(id="global-trend-graph"),
    
    # Household/Restaurant Prediction
    html.H2("Predict Household / Restaurant Food Waste"),
    html.Div([
        html.Label("Number of people / servings:"),
        dcc.Input(id="people-input", type="number", value=4, min=1),

        html.Label("Food Category:"),
        dcc.Dropdown(
            options=[{'label': f, 'value': f} for f in df["food_category"].unique()],
            value="Vegetables",
            id="food-category-input"
        ),

        html.Label("Avg Waste per Capita (Kg):"),
        dcc.Input(id="avg-waste-input", type="number", value=2.0, min=0),

        html.Label("Year:"),
        dcc.Input(id="year-input", type="number", value=2025, min=2000),

        html.Button("Predict Waste", id="predict-btn", n_clicks=0),
    ], style={'margin':'20px 0'}),

    html.Div(id="prediction-output", style={'fontWeight':'bold', 'fontSize':'18px'}),
    html.Div(id="recommendation-output", style={'marginTop':'10px', 'color':'green'}),
])

# -----------------------------
# Callbacks
# -----------------------------
@app.callback(
    Output("global-trend-graph", "figure"),
    Input("country-dropdown", "value"),
    Input("year-slider", "value")
)
def update_global_trends(selected_countries, selected_years):
    filtered_df = df[
        (df["country"].isin(selected_countries)) &
        (df["year"] >= selected_years[0]) &
        (df["year"] <= selected_years[1])
    ]
    if filtered_df.empty:
        fig = px.line(title="No data for selected filters")
    else:
        grouped = filtered_df.groupby(["year","country"])["total_waste_tons"].sum().reset_index()
        fig = px.line(grouped, x="year", y="total_waste_tons", color="country", title="Total Food Waste Over Time")
    return fig

@app.callback(
    Output("prediction-output","children"),
    Output("recommendation-output","children"),
    Input("predict-btn","n_clicks"),
    Input("people-input","value"),
    Input("food-category-input","value"),
    Input("avg-waste-input","value"),
    Input("year-input","value")
)
def predict_waste(n_clicks, people, food_cat, avg_waste, year):
    if n_clicks == 0:
        return "", ""
    
    # -----------------------------
    # Input Validation
    # -----------------------------
    if people is None or people <= 0:
        return "Error: Enter a valid number of people.", ""
    if avg_waste is None or avg_waste < 0:
        return "Error: Enter a valid Avg Waste per Capita.", ""
    if year is None or year < 2000:
        return "Error: Enter a valid year.", ""

    # Convert to millions for model
    population_millions = people / 1_000_000

    # Create DataFrame with proper values
    X_new = pd.DataFrame([[year, population_millions, avg_waste]], 
                         columns=["year", "population_millions", "avg_waste_per_capita"])
    
    # Predict using ML model
    predicted_waste = model.predict(X_new)[0]

    # Recommendations
    if predicted_waste > avg_waste * people:
        rec = html.Ul([
            html.Li("Redistribute surplus food to neighbors or charities"),
            html.Li("Use recipes that consume surplus ingredients"),
            html.Li("Track expiry dates and plan meals accordingly"),
            html.Li("Reduce portion sizes or monitor stock levels")
        ])
    else:
        rec = html.P("Waste is within acceptable range. Keep monitoring!")

    return f"Predicted Waste: {predicted_waste:.2f} Kg", rec

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)





