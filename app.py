import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import holidays

st.title("Vegetable Demand Forecast Dashboard")

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("demand_model.pkl")

# ----------------------------
# LOAD DATA
# ----------------------------
file = "PKM_Project_Dataset.xlsx"
sales = pd.read_excel(file, sheet_name="SALES DATA")

sales = sales.rename(columns={
    'Invoice Date':'ds',
    'SalQty':'y',
    'Material name':'product'
})

sales['ds'] = pd.to_datetime(sales['ds'])

# ----------------------------
# SELECT PRODUCT
# ----------------------------
product = st.selectbox(
    "Select Product",
    sales['product'].unique()
)

product_data = sales[sales['product']==product]

# ----------------------------
# TIME FEATURES
# ----------------------------
product_data['month'] = product_data['ds'].dt.month
product_data['weekday'] = product_data['ds'].dt.weekday
product_data['weekend'] = product_data['weekday'].apply(lambda x:1 if x>=5 else 0)
product_data['quarter'] = product_data['ds'].dt.quarter

# ----------------------------
# HOLIDAYS
# ----------------------------
india_holidays = holidays.India()

product_data['holiday'] = product_data['ds'].apply(
    lambda x:1 if x in india_holidays else 0
)

# ----------------------------
# WEDDING SEASON
# ----------------------------
wedding_months=[1,2,4,5,11,12]

product_data['wedding_season'] = product_data['month'].apply(
    lambda x:1 if x in wedding_months else 0
)

# ----------------------------
# LAG FEATURES
# ----------------------------
product_data['lag_1'] = product_data['y'].shift(1)
product_data['lag_3'] = product_data['y'].shift(3)
product_data['lag_7'] = product_data['y'].shift(7)
product_data['lag_14'] = product_data['y'].shift(14)
product_data['lag_30'] = product_data['y'].shift(30)

product_data['rolling_mean_7'] = product_data['y'].shift(1).rolling(7).mean()
product_data['rolling_mean_14'] = product_data['y'].shift(1).rolling(14).mean()

product_data = product_data.fillna(method='bfill').fillna(method='ffill')

# ----------------------------
# PRODUCT ENCODING
# ----------------------------
product_data['product_id'] = 0

# ----------------------------
# FEATURE LIST
# ----------------------------
features = [
'product_id',
'month',
'weekday',
'weekend',
'holiday',
'wedding_season',
'quarter',
'lag_1',
'lag_3',
'lag_7',
'lag_14',
'lag_30',
'rolling_mean_7',
'rolling_mean_14'
]

X = product_data[features]

# ----------------------------
# PREDICTION
# ----------------------------
pred = model.predict(X)

product_data['Predicted_Demand'] = pred

# ----------------------------
# VISUALIZATION
# ----------------------------
fig = px.line(
    product_data,
    x="ds",
    y=["y","Predicted_Demand"],
    title=f"{product} Demand Forecast",
)

st.plotly_chart(fig)

st.write("Actual vs Predicted Data")
st.dataframe(product_data[['ds','y','Predicted_Demand']])