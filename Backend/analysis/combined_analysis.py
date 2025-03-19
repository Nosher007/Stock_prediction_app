# analysis/combined_analysis.py
import pandas as pd
from data.data_ingestion import fetch_stock_data
from analysis.sentiment_analysis_newsapi import analyze_stock_news_sentiment
from Backend.config import NEWS_API_KEY  # Your API key stored in config.py


def combined_stock_and_sentiment_analysis(symbol: str, period: str = "1y"):
    # Fetch historical stock data for the given symbol
    stock_data = fetch_stock_data(symbol, period)

    # Fetch sentiment analysis data using NewsAPI
    sentiment_data = analyze_stock_news_sentiment(symbol, NEWS_API_KEY)

    # Convert the sentiment data to a DataFrame for easier viewing/manipulation
    sentiment_df = pd.DataFrame(sentiment_data)

    print("Stock Data Sample:")
    print(stock_data.head())
    print("\nSentiment Analysis Sample:")
    print(sentiment_df.head())


if __name__ == "__main__":
    combined_stock_and_sentiment_analysis("AAPL", "6mo")
