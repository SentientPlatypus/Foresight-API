from flask import Flask,render_template, request, session, redirect, url_for
import ssl
from threading import Thread
import pandas as pd
import requests
import yfinance as yf
from flaskApp import constants
from bs4 import BeautifulSoup
import tensorflow as tf
from flaskApp.scraper import *
from flask_cors import CORS, cross_origin
import numpy as np
import yfinance as yf
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

context = ssl.create_default_context()


def createApp():
    app = Flask(
    __name__,
    template_folder=r"templates",
    static_folder=r"static"
    )
    return app
app = createApp()

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/isTickerValid/<string:ticker>")
@cross_origin()
def isTickerValid(ticker:str) -> str:
    """Checks if ticker is valid."""
    data = requests.get(f"{constants.GOOGLE_FINANCE_URL}{ticker}", headers=constants.REQ_HEADER).text
    soup = BeautifulSoup(data, "lxml")
    if soup.find("ul", {"class":constants.OPTIONS_LIST_CLASSES}):
        return constants.TRUE
    else:
        return constants.TRUE if soup.find("div", {"class":"zzDege"}) else constants.FALSE

@app.route("/getInfo/<string:ticker>")
@cross_origin()
def getInfo(ticker:str) -> dict:
    """Prerequisite is that ticker must be valid. Use isTickerValid for this."""
    scrapingURL = getScrapingURL(ticker)
    data = requests.get(scrapingURL, headers=constants.REQ_HEADER).text
    soup = BeautifulSoup(data, "lxml")
    
    info_we_need = {
        "companyName" : scrapeCompanyName(soup),
        "currentValue" : {
            "value" : scrapePrice(soup),
            "change" : getPriceChangeStr(getFloat(scrapePrice(soup)), getFloat(scrapePrevClose(soup)), "Today")
        },
        "marketStatus" : scrapeMarketStatus(soup),
        "companyDesc" : scrapeCompanyDesc(soup),
        "companyLogoUrl" : scrapeCompanyLogo(scrapeCompanyWebsite(soup))
    }
    return info_we_need

@app.route("/getFinancials/<string:ticker>")
@cross_origin()
def getFinancials(ticker:str) -> dict:
    scrapingURL = getScrapingURL(ticker)
    print(scrapingURL)
    data = requests.get(scrapingURL, headers=constants.REQ_HEADER).text
    soup = BeautifulSoup(data, "lxml")
    financials = {
        "incomeStatement": scrapeIncomeStatement(soup),
        "balanceSheet":scrapeBalanceSheet(soup),
        "cashFlow":scrapeCashFlow(soup)
    }
    return financials


@app.route("/getNumbers/<string:ticker>")
@cross_origin()
def getNumbers(ticker:str):
    toDisplay:pd.DataFrame = yf.download(ticker, period="max", progress=False)

    df:pd.DataFrame = yf.download(ticker, period="6mo", progress=False)
    y = df['Close'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 90  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan

    results:pd.DataFrame = df_past.append(df_future).set_index('Date')
    results = results[results["Forecast"].notna()]
    results["Open"] = results["Forecast"]
    results["High"] = results["Forecast"]
    results["Low"] = results["Forecast"]
    resultsShifted = results.shift(-1)
    results["Close"] = resultsShifted["Open"]
    final = pd.concat([toDisplay, results], sort=False, join="inner")
    csv_df = final.to_csv()
    newStr = str(csv_df)
    return newStr[24:]



@app.route("/getNews/<string:ticker>")
@cross_origin()
def getNews(ticker:str) -> dict:
    scrapingURL = getScrapingURL(ticker=ticker)
    data = requests.get(scrapingURL, headers=constants.REQ_HEADER).text
    soup = BeautifulSoup(data, "lxml")
    return scrapeNews(soup)

@app.route("/")
@cross_origin()
def home():
    return {"stocks":0, "maidens":0}


if __name__ == '__main__':
    def run():
        app.run(host='0.0.0.0',port=8080)
    def keep_alive():
        t = Thread(target=run)
        t.start()
    keep_alive()

