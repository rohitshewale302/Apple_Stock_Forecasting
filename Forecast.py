import pandas as pd
import streamlit as st
from pickle import load
import statsmodels.api as sm
import matplotlib.pyplot as plt
data_close = load(open('data_close.sav','rb'))

st.title('Model Deployment: Apple Stock Forecasting')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.businessinsiderbd.com/media/imgAll/2020October/en/apple-logo-2211110659.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url() 

periods = st.number_input('Number of Days',min_value=1)

datetime = pd.DataFrame(pd.date_range('2020-01-01', periods=periods,freq='B'), columns = ['Date'])

model_sarima_final = sm.tsa.SARIMAX(data_close.Close,order=(2,1,0),seasonal_order=(1,1,0,66)).fit()
forecast = pd.DataFrame(model_sarima_final.predict(len(data_close),len(data_close)+periods-1))
forecast.columns = ['Close']

data_forecast = forecast.set_index(datetime.Date)
st.write(data_forecast)

fig,ax = plt.subplots()
ax.plot(data_close['Close'], label = 'Close')
ax.plot(data_forecast, label = 'Forecast')
ax.set_title('Apple Stock Forecast')
ax.set_xlabel('Year')
ax.set_ylabel('Stock Price')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True)
st.pyplot(fig)

fig,ax = plt.subplots()
ax.plot(data_forecast, label = 'Forecast')
ax.set_title('Apple Stock Forecast')
ax.set_xlabel('Year')
ax.set_ylabel('Stock Price')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True)
st.pyplot(fig)
                    



