# api/routers.py
from flask import Blueprint, request, jsonify
from Backend.analysis.combined_analysis import get_combined_analysis

api = Blueprint('api', __name__)


@api.route('/hellos', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello World'})


@api.route('/combined-analysis', methods=['GET'])
def combined_analysis():
    """
    Endpoint to retrieve combined stock data and sentiment analysis.
    Query parameters:
      - symbol: Stock ticker symbol (default "AAPL")
      - period: Time period for stock data (default "1y")
    """
    symbol = request.args.get("symbol", "AAPL")
    period = request.args.get("period", "1y")

    result = get_combined_analysis(symbol, period)
    return jsonify(result)
