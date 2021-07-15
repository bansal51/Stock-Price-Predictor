import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

# load dataset from Yahoo Finance
data = pd.read_csv("AMZN.csv")
print(data.head())

# Printing out graph of Amazon close prices by date
close = data['Close']
ax = close.plot(title = 'Amazon')
ax.set_xlabel('Date')
ax.set_ylabel('Close')
plt.show()

# we only need the date and close tabs, creating a new dataframe 
data["Date"] = pd.to_datetime(data["Date"], infer_datetime_format=True)
data = data[["Date", "Close"]]

# We are using the Facebook Prophet model --> need to rename columns
data = data.rename(columns={"Date" : "ds", "Close" : "y"})

# Predict Stock Trend of Amazon using Facebook Prophet
model = Prophet()
model.fit(data)

predict = model.make_future_dataframe(periods=365)
forcast = model.predict(predict)
print(forcast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

# Graph the results
graph = model.plot(forcast, xlabel="Date", ylabel="Price")
plt.show()