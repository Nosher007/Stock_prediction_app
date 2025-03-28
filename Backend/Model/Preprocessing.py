# Backend/Model/Preprocessing.py
import json
import pandas as pd
import os
import ast


def load_json_data(json_filepath: str) -> dict:
    """
    Load the combined data from the JSON file.
    """
    with open(json_filepath, "r", encoding="utf-8") as f:
        combined_data = json.load(f)
    return combined_data


def prepare_stock_df(stock_data: list) -> pd.DataFrame:
    """
    Convert the stock_data list from JSON into a DataFrame.
    """
    stock_df = pd.DataFrame(stock_data)
    # Convert the Date column to datetime and then to a string in ISO format
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], utc=True).dt.date.astype(str)
    return stock_df


def prepare_sentiment_df(sentiment_data: list) -> pd.DataFrame:
    """
    Convert the sentiment_data list from JSON into a DataFrame.
    Extract the 'compound' score from the sentiment column if necessary.
    """
    sentiment_df = pd.DataFrame(sentiment_data)
    # Convert publishedAt to datetime with utc and then to date, and then to string
    sentiment_df["publishedAt"] = pd.to_datetime(sentiment_df["publishedAt"], utc=True).dt.date.astype(str)

    # In case the 'sentiment' column is a string representation of a dict,
    # convert it to a dictionary and extract 'compound'
    def parse_sentiment(s):
        try:
            if isinstance(s, str):
                d = ast.literal_eval(s)
                return d.get("compound", 0.0)
            elif isinstance(s, dict):
                return s.get("compound", 0.0)
        except Exception:
            return 0.0

    sentiment_df["compound"] = sentiment_df["sentiment"].apply(parse_sentiment)

    # Rename publishedAt to Date for merging
    sentiment_df.rename(columns={"publishedAt": "Date"}, inplace=True)
    return sentiment_df


def merge_datasets(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment by date and merge stock and sentiment data on Date.
    """
    # Aggregate sentiment data: average the compound score per day
    sentiment_agg = (
        sentiment_df.groupby("Date")["compound"]
        .mean()
        .reset_index()
        .rename(columns={"compound": "avg_compound_sentiment"})
    )

    # Ensure the Date columns in both DataFrames are of the same type (string)
    stock_df["Date"] = stock_df["Date"].astype(str)
    sentiment_agg["Date"] = sentiment_agg["Date"].astype(str)

    # Merge on Date (left join to retain all stock dates)
    merged_df = pd.merge(stock_df, sentiment_agg, on="Date", how="left")
    merged_df["avg_compound_sentiment"] = merged_df["avg_compound_sentiment"].fillna(0.0)
    return merged_df


def save_csv(df: pd.DataFrame, filename: str, output_dir: str = "."):
    """
    Save a DataFrame as a CSV file.
    """
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Merged data saved to: {output_path}")


def main():
    # Adjust the path to your JSON file as needed
    json_filepath = "../analysis/combined_analysis.json"
    output_dir = "."

    # Load JSON data
    combined_data = load_json_data(json_filepath)

    # Create DataFrames for stock_data and sentiment_data
    stock_df = prepare_stock_df(combined_data.get("stock_data", []))
    sentiment_df = prepare_sentiment_df(combined_data.get("sentiment_data", []))

    print("Stock DataFrame:")
    print(stock_df.head())
    print("\nSentiment DataFrame:")
    print(sentiment_df.head())

    # Merge the two DataFrames on Date
    merged_df = merge_datasets(stock_df, sentiment_df)

    print("\nMerged DataFrame:")
    print(merged_df.head())

    # Save the merged DataFrame as a CSV file
    save_csv(merged_df, "merged_data.csv", output_dir)


if __name__ == "__main__":
    main()
