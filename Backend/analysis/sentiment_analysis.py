import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Backend.Config.config import NEWS_API_KEY

def fetch_stock_news_newsapi(symbol: str, api_key: str, page_size: int = 100):
    """
    Fetch news headlines for the given stock symbol using NewsAPI.

    Parameters:
      - symbol: Stock ticker or relevant keyword (e.g., "AAPL").
      - api_key: Your NewsAPI API key.
      - page_size: Number of articles to retrieve (default is 10).

    Returns:
      A list of news articles (dictionaries) from NewsAPI.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "apiKey": api_key,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data.get("status") != "ok":
        return []
    return data.get("articles", [])

def analyze_sentiment(headline:str,analyzer:SentimentIntensityAnalyzer):
    return analyzer.polarity_scores(headline)

def analyze_stock_news_sentiment(symbol:str,api_key: str, page_size: int = 10):
    articles = fetch_stock_news_newsapi(symbol, api_key, page_size)
    analyzer = SentimentIntensityAnalyzer()
    sentiment_results = []

    for article in articles:
        headline = article.get("title", "")
        sentiment = analyze_sentiment(headline, analyzer)
        sentiment_results.append({
            "title": headline,
            "description": article.get("description", ""),
            "url": article.get("url", ""),
            "publishedAt": article.get("publishedAt", ""),
            "sentiment": sentiment
        })
    return sentiment_results

if __name__ == '__main__':
    symbol = "AAPL"
    sentiments = analyze_stock_news_sentiment(symbol, NEWS_API_KEY)
    for item in sentiments:
        print(item)
