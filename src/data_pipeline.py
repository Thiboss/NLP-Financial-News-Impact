from .utils import load_json, save_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pandas as pd
from .news_fetcher import NewsFetcher
from .yahoo_fetcher import StockDataFetcher
from .sentiment_analyser import SentimentAnalyzer
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np


class Pipeline:
    """
    Class representing the pipeline for our data & model:

    I. Model Training Pipeline : 'use_pretrained_model' = False

        If the argument 'use_pretrained_model' is set to False, the pipeline will run the following steps to train a new model:

        If the argument 'use_cache' is set to True, the pipeline will use cached data stored in the 'data' directory
        (i.e., the articles & sentiment analysis results + stock data from Yahoo Finance).

        If the argument 'use_cache' is set to False, the pipeline will fetch new data as follows:
            1. Fetching and saving stock data from Yahoo Finance.
            2. Fetching and saving articles from a news API.
            3. Performing sentiment analysis on the articles.

        The pipeline can then be used to fit the model on the processed data:
            1. Load all the data (news, sentiment and stock returns).
            2. Create features from the sentiment data.
            3. Prepare the data for the model (train/test split).
            4. Train the model.
            5. Evaluate the model.

    II. Model Inference Pipeline : 'use_pretrained_model' = True

        If the argument 'use_pretrained_model' is set to True, the pipeline will load a pre-trained model
        and use it to make predictions on new data that is fetched and processed in the same way as above.
    """

    def __init__(self,
                 use_pretrained_model: bool = False,
                 use_cache: bool = True,
                 sentiment_path: str = "data/articles/articles_with_sentiment.json",
                 returns_path: str = "data/yahoo_finance/yahoo_data.xlsx",
                 pretrained_model_name: str = "pretrained_model.pkl"):
        """
        Parameters:
            use_pretrained_model (bool): If True, use a pre-trained model for inference.
            use_cache (bool): If True, use cached data. If False, fetch new data.
            sentiment_path (str): Path to the sentiment data JSON file.
            returns_path (str): Path to the stock returns data Excel file.
            model_used: The machine learning model to be used in the pipeline.
        """
        self.use_pretrained_model = use_pretrained_model
        self.use_cache = use_cache
        self.sentiment_path = sentiment_path
        self.returns_path = returns_path
        self.pretrained_model_name = pretrained_model_name + ".pkl"

        self.model = None

    def _load_data(self, specific_search_terms: list = None) -> Tuple[Dict, pd.DataFrame]:
        """
        Load all the data needed for the model pipeline.

        If 'use_cache' is True, it loads the data from the specified paths.
        Else, it fetches new data from Yahoo Finance and a news API, and performs sentiment analysis on the articles.
        """
        if self.use_cache:
            sentiment_dict = load_json(self.sentiment_path)
            price_df = pd.read_excel(self.returns_path, index_col=0, parse_dates=True)
        else:
            articles = NewsFetcher(specific_search_terms).fetch_articles(from_date=self.start_date, to_date=self.end_date, max_pages=5)
            sentiment_dict = SentimentAnalyzer().run(articles)
            if specific_search_terms is None:
                tickers = load_json("config.json")["INDEX_MAPPING"].values()
            else:
                tickers = []
                for key, value in load_json("config.json")["INDEX_MAPPING"].items():
                    if key in specific_search_terms:
                        tickers.append(value)
            price_df = StockDataFetcher().fetch_and_save(tickers=tickers,
                                                         start_date=self.start_date,
                                                         end_date=self.end_date)
        sentiment_df = pd.DataFrame([
            {
                "search_term": article.get("search_term"),
                "date": article.get("published_at"),
                "sentiment": article.get("sentiment")
            }
            for article in sentiment_dict
        ])

        return sentiment_df, price_df
    
    def _create_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from sentiment data analysis.
        This method aggregates sentiment data by search term and date, computes proportions,
        and derives additional features such as sentiment score and volume.

        Parameters:
            sentiment_df (pd.DataFrame): DataFrame containing sentiment data with columns 'search_term', 'date', and 'sentiment'.

        Returns:
            pd.DataFrame: A DataFrame with aggregated sentiment data, including proportions and sentiment scores.
        """
        sentiment_pivot : pd.DataFrame = (
            sentiment_df
            .groupby(["search_term", "date"])["sentiment"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )

        sentiment_pivot["total"] = sentiment_pivot[["positive", "negative", "neutral"]].sum(axis=1)
        for col in ["positive", "negative", "neutral"]:
            sentiment_pivot[f"p_{col[:3]}"] = sentiment_pivot[col] / sentiment_pivot["total"]
        sentiment_pivot["score_net"] = sentiment_pivot["p_pos"] - sentiment_pivot["p_neg"]
        sentiment_pivot["volume"] = sentiment_pivot["total"]
        sentiment_pivot["ticker"] = sentiment_pivot["search_term"].map(load_json("config.json")["INDEX_MAPPING"])

        return sentiment_pivot

    def _prepare_dataset(self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the dataset by merging features and target, and splitting into train and test sets if not using a pre-trained model.

        Parameters:
            price_df (pd.DataFrame): DataFrame containing stock price data
            sentiment_df (pd.DataFrame): DataFrame containing sentiment features with columns 'date', 'ticker', and sentiment scores.
        """
        # Transform prices to returns
        returns_df = price_df.pct_change().iloc[1:, :]
        
        # Convert sentiment_df date column to datetime for proper comparison
        sentiment_df = sentiment_df.copy()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], format='%d/%m/%Y')
        
        # Create a list to store all the formatted data
        formatted_data = []
        
        # For each row in sentiment_df, get the 10 preceding returns for that ticker and date
        for idx, row in sentiment_df.iterrows():
            ticker = row['ticker']
            date = row['date']
            
            # Get the target return (return on the date of the article)
            target_return = returns_df[ticker].get(date, np.nan)
            if pd.isna(target_return):
                # If no return data for this exact date, skip this row
                continue
            
            # Get returns for this ticker up to the date (excluding the date itself)
            ticker_returns = returns_df[ticker][returns_df.index < date]
            
            # Get the last 10 returns (or less if not enough data available)
            last_10_returns = ticker_returns.tail(10)
            
            # If we have less than 10 returns, pad with zeros at the beginning
            returns_list = last_10_returns.values.tolist()
            while len(returns_list) < 10:
                returns_list.insert(0, 0.0)
            
            # Replace NaN values with 0
            returns_list = [0.0 if pd.isna(val) else val for val in returns_list]
            
            # Add this data to our formatted data
            row_data = row.to_dict()
            row_data['target_return'] = target_return
            for i, return_val in enumerate(returns_list):
                row_data[f'return_{i+1}'] = return_val
            
            formatted_data.append(row_data)
        
        formatted_data = pd.DataFrame(formatted_data)

        # Split into features and target
        # Target is the return on the date of the article (target_return) converted to binary classification (1 if return > 0, else 0)
        # Features are the sentiment features and the 10 preceding returns (return_0 to return_9)
        target = (formatted_data['target_return'] > 0).astype(int)
        features = formatted_data.drop(columns=['search_term', 'ticker', 'date', 'target_return'])

        return features, target
    
    def _data_pipeline(self, specific_search_terms: list = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run the data pipeline to prepare features and target for model training or inference.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame and the target Series.
        """
        sentiment_df, returns_df = self._load_data(specific_search_terms)
        sentiment_df = self._create_features(sentiment_df)
        features, target = self._prepare_dataset(returns_df, sentiment_df)
        
        return features, target

    def fit(self, start_date: str, end_date: str, save_model: bool = True, model_analysis=False):
        """
        Trains a new model based on the provided data.
        It then provides analysis of the model performance and saves the model if 'save_model' is True.

        Parameters:
            start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
            end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.
            save_model (bool): If True, saves the trained model to a file.
        """
        self.start_date = start_date
        self.end_date = end_date

        features, target = self._data_pipeline()

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model = model.fit(X_train, y_train)

        if save_model:
            joblib.dump(self.model, os.path.join("data/pretrained_models", self.pretrained_model_name))

        if model_analysis:
            self._model_analysis(X_test, y_test)
    
    def _model_analysis(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Analyze the model performance using classification report, confusion matrix, and SHAP values.
        Also plots feature importances and confusion matrix.

        Parameters:
            X_test (pd.DataFrame): The test features.
            y_test (pd.Series): The true labels for the test set.
        """

        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.show()

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_names = X_test.columns
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(8, 5))
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()

    def load_pretrained_model(self, pretrained_model_name: str):
        """
        Load a pre-trained model from the specified path.

        Parameters:
            pretrained_model_name (str): The name of the pre-trained model file to load.
        """
        self.model = joblib.load(os.path.join("data/pretrained_models", pretrained_model_name))

    def predict(self, date: str, ticker: str):
        """
        Run the model for a specific ticker and date range to make predictions.

        Parameters:
            date (str): The date for which to make predictions in 'YYYY-MM-DD' format.
            ticker (str): The stock ticker for which to make predictions.
        """
        # Parameters
        self.use_cache = False
        self.end_date = date
        self.start_date = (pd.to_datetime(date) - pd.DateOffset(days=20)).strftime('%Y-%m-%d')
        
        # Get the index name from the ticker with the config mapping
        index_mapping = load_json("config.json")["INDEX_MAPPING"]
        if ticker not in index_mapping.values():
            raise ValueError(f"Ticker '{ticker}' not found in the index mapping.")
        index_name = next((k for k, v in index_mapping.items() if v == ticker), None)

        # Run the data pipeline to get features
        features, _ = self._data_pipeline([index_name])

        # Predict using the model
        res = self.model.predict(features.head(1))[0]

        print(f"Prediction for {ticker} on {date}: {'Positive' if res == 1 else 'Negative'}")
