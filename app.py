import streamlit as st
import joblib
import numpy as np
import pandas as pd

model    = joblib.load('model.pkl')
features = joblib.load('features.pkl')

st.title("House Price Predictor")
st.write("Enter the details of a house to get an estimated sale price.")

col1, col2 = st.columns(2)

with col1:
    overall_qual   = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area    = st.number_input("Living Area (sq ft)", 500, 6000, 1500)
    garage_cars    = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4])
    total_bsmt_sf  = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
    first_flr_sf   = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000)

with col2:
    full_bath      = st.selectbox("Full Bathrooms", [1, 2, 3, 4])
    totrms_abvgrd  = st.slider("Total Rooms (above ground)", 2, 14, 6)
    year_built     = st.number_input("Year Built", 1872, 2010, 1990)
    year_remod     = st.number_input("Year Remodelled", 1950, 2010, 2000)
    fireplaces     = st.selectbox("Fireplaces", [0, 1, 2, 3])
    
if st.button("Predict Price"):
    input_data = pd.DataFrame([[
        overall_qual, gr_liv_area, garage_cars, total_bsmt_sf,
        first_flr_sf, full_bath, totrms_abvgrd, year_built,
        year_remod, fireplaces
    ]], columns=features)

    log_price = model.predict(input_data)[0]
    price = np.expm1(log_price)

    st.success(f"Estimated Sale Price: ${price:,.0f}")
    st.caption("Based on Ames Housing Dataset (Iowa, USA)")
    