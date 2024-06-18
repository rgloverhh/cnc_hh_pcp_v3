import numpy as np
import streamlit as st
import pickle
from xgboost import XGBRegressor

st.title("CNC Primary Care Service Level Predictor")

def load_model():
    with open('pcp_xg_model.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

model = load_model()

def prediction(input1, input2, input3, input4):
    sl = model.predict([[input1, input2, input3, input4]])
    return sl[0]


#inputs - calls offered, AHT, not ready rate, total ftes, fte callouts

st.text("Please fill in the responses below to predict primary care service level")

calls_offered = st.number_input(label="Enter a call volume between 500 and 3000", min_value=500, max_value=4000, step=10, value="")
aht = st.number_input(label="Average Handle Time (in decimal format, i.e. 5.5 = 5min 30sec -> 0.1 = 6 sec)", min_value=4.0, max_value=7.0, step=0.1, value="")
not_ready = st.number_input(label="Not Ready Rate (%)", min_value=15.0, max_value=35.0, step=0.5, value="")
ftes_logged_in = st.number_input(label="Choose the total number of FTEs logged in for the day (use PowerBI CNC Call Metrics Staffing as a guide)", min_value=20.0, max_value=40.0, step=0.5, value="")

not_ready_con = not_ready/100

sl_prediction_temp = prediction(calls_offered, aht, not_ready_con, ftes_logged_in)
sl_prediction = round((sl_prediction_temp*100),1)

st.header("Service Level Prediction")
if sl_prediction <= 0:
    st.subheader("0%")
elif sl_prediction >= 100:
    st.subheader("100%")
else:
    st.subheader(f"{sl_prediction}%")

st.caption("Modeling used is eXtreme Gradient Boosting (XGBoost) and was trained on CNC call data Oct 3 2022 - Jun 14 2024 (as of Jun 17 2024)")