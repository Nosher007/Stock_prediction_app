# analysis/combined_analysis.py
import pandas as pd
import json
from Backend.analysis.data.data_ingestion import fetch_stocks_data
from Backend.analysis.sentiment_analysis import analyze_stock_news_sentiment
from Backend.Config.config import NEWS_API_KEY


def convert_timestamps(obj):
    """
    Recursively converts Pandas Timestamp objects in the given object
    (which may be a dict, list, etc.) into ISO formatted strings.
    """
    if isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


def get_combined_analysis(symbol: str, period: str = "1y"):
    """
    Fetches historical stock data and sentiment analysis, returning a combined result as a dictionary.
    """
    # Fetch historical stock data for the given symbol
    stock_data = fetch_stocks_data(symbol, period)

    # Convert the Date column (if present) to string format for JSON serialization
    if not stock_data.empty and "Date" in stock_data.columns:
        stock_data["Date"] = stock_data["Date"].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)

    # Fetch sentiment analysis data using NewsAPI
    sentiment_data = analyze_stock_news_sentiment(symbol, NEWS_API_KEY)

    # Convert stock data to a list of dictionaries for JSON serialization
    stock_records = stock_data.to_dict(orient="records") if not stock_data.empty else []

    # Combine the results into one dictionary
    result = {
        "symbol": symbol,
        "period": period.strip(),
        "stock_data": stock_records,
        "sentiment_data": sentiment_data
    }

    # Convert any remaining Timestamps recursively
    result = convert_timestamps(result)
    return result


if __name__ == "__main__":
    # For local testing: get combined analysis result and print it
    result = get_combined_analysis("AAPL", "15y")
    print(result)

    # Save the result dictionary as a JSON file
    with open("combined_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print("Combined analysis saved to combined_analysis.json")
