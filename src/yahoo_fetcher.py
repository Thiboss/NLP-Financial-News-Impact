from datetime import datetime
from typing import List
import yfinance as yf
import pandas as pd
import os


class StockDataFetcher:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_and_save(self, tickers: List[str], start_date: str, end_date: str):
        """
        Fetches stock data for given tickers and saves them to excel file.

        Parameters:
            tickers (List[str]): List of stock tickers to fetch data for.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD'
        """
        closing_prices = pd.DataFrame()
        for ticker in tickers:
            print(f"➡️  Downloading {ticker} from {start_date} to {end_date}")
            df: pd.DataFrame = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=False
            )
            close_column = df['Close'].copy()
            close_column.name = ticker
            closing_prices[ticker] = close_column

        if not closing_prices.empty:
            path = os.path.join(self.output_dir, "yahoo_data.xlsx")
            closing_prices.to_excel(path)

        return closing_prices