import numpy as np
import pickle
import pandas as pd
import streamlit as st 


pickle_in_lr = open("clf_gini.pkl","rb")
lr_model=pickle.load(pickle_in_lr)
pickle_in_scaler = open("scaler.pkl","rb")
scaler=pickle.load(pickle_in_scaler)


def predict_aqi(T,TM,Tm,SLP,H,VV,V,VM):
    test_values = np.array([[T,TM,Tm,SLP,H,VV,V,VM]])
    test_values_scaled=scaler.transform(test_values)
    prediction=lr_model.predict(test_values_scaled)
    return prediction



def main():
    st.title("Air Quality Index Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Admission Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    T = st.text_input("Average Temperature in Celcius","Type Here")
    TM = st.text_input("TM Maximum temperature (°C)")
    Tm = st.text_input("Tm Minimum temperature (°C)")
    SLP = st.text_input("Atmospheric pressure at sea level (hPa)")
    H=st.text_input("Average relative humidity (%")
    VV  = st.text_input("VV Average visibility (Km)")
    V = st.text_input("CGPA (out of 10V Average wind speed (Km/h))")
    VM = st.text_input("VM Maximum sustained wind speed (Km/h)")
    result=""
    if st.button("Predict"):
        result=predict_aqi(T,TM,Tm,SLP,H,VV,V,VM)
    st.success('The admission chances are {}%'.format(result))
    if st.button("About"):
        st.text("Predicting AQI")
        st.text("By I074 Karan Bedi, I073 Kapil Goya I072 Yashvi Shah")

if __name__=='__main__':
    main()
