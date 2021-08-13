import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
import requests
import time
import datetime

today = datetime.datetime.today()
stock = 'CRM'
period1 = int(time.mktime(datetime.datetime(today.year - 1, today.month, today.day, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(today.year, today.month, today.day, 23, 59).timetuple()))
interval = '1d' # 1d, 1m

csv_url = f'https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

# load dataset from Yahoo Finance
data = pd.read_csv(csv_url)
print(data.head())

# Printing out graph of Salesforce close prices by date
close = data['Close']
ax = close.plot(title = 'Salesforce')
ax.set_xlabel('Date')
ax.set_ylabel('Close')
plt.show()

# we only need the date and close tabs, creating a new dataframe 
data["Date"] = pd.to_datetime(data["Date"], infer_datetime_format=True)
data = data[["Date", "Close"]]

# We are using the Facebook Prophet model --> need to rename columns
data = data.rename(columns={"Date" : "ds", "Close" : "y"})

# Predict Stock Trend of Salesforce using Facebook Prophet
model = Prophet()
model.fit(data)

predict = model.make_future_dataframe(periods=365)
forcast = model.predict(predict)
print(forcast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

# Graph the results
graph = model.plot(forcast, xlabel="Date", ylabel="Price")
plt.show()