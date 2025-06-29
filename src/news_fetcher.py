from src.utils import load_json, save_json
from newsapi import NewsApiClient
from datetime import datetime


class NewsFetcher:
    """
    Module in charge of loading financial news from NewsAPI based on a configuration file.
    Articles are retrieved from the API, pre-processed and stored in a JSON file to be used.
    """

    def __init__(self, specific_search_terms: list = None):
        self.config = load_json("config.json")["NEWS_API"]
        self.api_key = self.config["NEWS_API_KEY"]
        self.language = self.config["LANGUAGE"]
        self.sort_by = self.config["SORT_BY"]
        self.search_terms = self.config["SEARCH_TERMS"] if specific_search_terms is None else specific_search_terms
        self.client = NewsApiClient(api_key=self.api_key)

    def fetch_articles(self, from_date: str, to_date: str, max_pages: int = 5) -> list:
        """
        Fetches articles from NewsAPI based on date range.

        Parameters:
            from_date (str): Start date in YYYY-MM-DD format.
            to_date (str): End date in YYYY-MM-DD format.
            max_pages (int): Maximum number of pages to fetch for each search term.

        Returns:
            list: A list of articles with metadata.
        """
        all_articles = []
        for term in self.search_terms:
            for page in range(1, max_pages + 1):
                try:
                    response = self.client.get_everything(
                        q=term,
                        from_param=from_date,
                        to=to_date,
                        language=self.language,
                        sort_by=self.sort_by,
                        page=page,
                        page_size=100
                    )
                    articles = response.get("articles", [])
                    if not articles:
                        break
                    for article in articles:
                        published_at = article.get("publishedAt")
                        if published_at:
                            try:
                                published_at = datetime.strptime(published_at[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
                            except Exception:
                                pass
                        all_articles.append({
                            "search_term": term,
                            "source": article["source"]["name"],
                            "author": article.get("author"),
                            "title": article.get("title"),
                            "description": article.get("description"),
                            "content": article.get("content"),
                            "url": article.get("url"),
                            "published_at": published_at
                        })
                except Exception as e:
                    print(f"[ERROR] Term: {term}, Page: {page}, Error: {e}")
                    break
        return all_articles
