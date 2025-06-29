from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .utils import save_json
import torch.nn.functional as F
from typing import List
from tqdm import tqdm
import torch


class SentimentAnalyzer:
    """
    A sentiment analysis class using a pre-trained model from Hugging Face's Transformers library.
    It uses the FinBERT model, which is specifically trained for financial sentiment analysis.
    """

    def __init__(self, model_name="ProsusAI/finbert", sentiment_path="data/articles/articles_with_sentiment.json"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels = ["negative", "neutral", "positive"]
        self.sentiment_path = sentiment_path

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predicts the sentiment of a list of texts.

        Parameters:
            texts (List[str]): A list of strings to analyze.

        Returns:
            List[str]: A list of predicted sentiments corresponding to the input texts.
        """
        predictions = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                label_idx = torch.argmax(probs, dim=1).item()
                predictions.append(self.labels[label_idx])
        return predictions

    def run(self, articles: List[dict]) -> List[dict]:
        """
        Runs sentiment analysis on a list of articles.
        Each article should be a dictionary containing "title" and "description".

        Parameters:
            articles (List[dict]): A list of articles where each article is a dictionary.

        Returns:
            List[dict]: The input articles with an added "sentiment" key for each article.
        """
        for article in tqdm(articles):
            if "title" not in article:
                pass
            
            if "description" not in article or article["description"] is None:
                desc = ""
            else:
                desc = article["description"]

            text: str = article["title"] + " " + desc
            article["sentiment"] = self.predict([text.strip()])[0]

        save_json(articles, self.sentiment_path)
        return articles
