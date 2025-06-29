import json

import pandas as pd
import random

import matplotlib.pyplot as plt

def finbert_analysis(sentiment_analysis_path: str = "data/articles/articles_with_sentiment.json"):
    """
    Performs a high-level analysis of FinBERT sentiment analysis results.
    This function:
    - Loads a JSON file containing a list of article dictionaries with sentiment annotations.
    - Plots the proportion of each sentiment (positive, negative, etc.) over time (by publication date).
    - Aggregates and displays a DataFrame grouped by article title and sentiment.
    - Prints a few random examples showing the article title, description, and sentiment.

    Parameters:
        sentiment_analysis_path (str): Path to the JSON file containing sentiment analysis results. 
                                       Defaults to "data/articles/articles_with_sentiment.json".
    Returns:
        None. Displays plots and prints DataFrame and sample articles to the console.
    """
    with open(sentiment_analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['published_at'] = pd.to_datetime(df['published_at'], dayfirst=True, errors='coerce')

    sentiment_by_date = df.groupby([df['published_at'].dt.date, 'sentiment']).size().unstack(fill_value=0)
    sentiment_by_date_pct = sentiment_by_date.div(sentiment_by_date.sum(axis=1), axis=0)

    sentiment_by_date_pct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='coolwarm')
    plt.title("Sentiment proportion by publication date")
    plt.xlabel("Publication Date")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()

    agg_df = df.groupby(['search_term', 'sentiment']).size().unstack(fill_value=0)
    print("Aggregated data :")
    print(agg_df.head(10))

    print("\nArticle examples :")
    samples = df.sample(min(3, len(df)))
    for _, row in samples.iterrows():
        print(f"Title : {row['title']}\nDescription : {row['description']}\nSentiment : {row['sentiment']}\n---")