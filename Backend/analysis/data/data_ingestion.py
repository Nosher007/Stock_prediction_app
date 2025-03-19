import yfinance as yf
import pandas as pd

def fetch_stocks_data(symbol:str,period:str='1y'):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    stocks = fetch_stocks_data('AAPL','1y')
    print(stocks)
