from pickle import TRUE
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START="2019-01-01"
TODAY="2022-01-01"

st.title("Stock Analysis of MAANG Companies")
stocks=("META","AAPL","AMZN","NFLX","GOOG")
selected_stock=st.selectbox("select dataset",stocks)

n_years=st.slider("Years for prediction:",1,4)
period=n_years*365

@st.cache
def load_data(ticker): 
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("Load data...")
data=load_data(selected_stock)
data_load_state.text("Loading data...done")

st.subheader('Raw Data')
st.write(data.tail())

st.subheader('To Buy or Not to Buy')
selected_stock=str(selected_stock)
sc=yf.Ticker(selected_stock)
st.write(sc.recommendations.tail())

def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#forecasting

df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

m=Prophet()
m.fit(df_train)

future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('forecast')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2=m.plot_components(forecast)
st.write(fig2)
