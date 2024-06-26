import polars as pl
from typing import List, Dict, Any
import aiohttp
import asyncio
from transformers import pipeline
import spacy
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
import re

class AdvancedNLPEventDetector:
    def __init__(self, api_key: str, sentiment_model: str = "finiteautomata/bertweet-base-sentiment-analysis"):
        self.api_key = api_key
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        self.nlp = spacy.load("en_core_web_sm")
        self.lda_model = None
        self.dictionary = None
        self.news_buffer = pl.DataFrame(schema={
            "title": pl.Utf8,
            "description": pl.Utf8,
            "source": pl.Utf8,
            "published_at": pl.Datetime,
            "sentiment": pl.Float32,
            "relevance_score": pl.Float32,
            "entities": pl.List(pl.Struct({"text": pl.Utf8, "label": pl.Utf8})),
            "topics": pl.List(pl.Struct({"topic_id": pl.UInt32, "score": pl.Float32}))
        })

    async def fetch_news(self, keywords: List[str]):
        async with aiohttp.ClientSession() as session:
            params = {
                "apiKey": self.api_key,
                "q": " OR ".join(keywords),
                "language": "en",
                "sortBy": "publishedAt"
            }
            async with session.get("https://newsapi.org/v2/everything", params=params) as response:
                data = await response.json()
                return data.get("articles", [])

    def process_news(self, articles: List[Dict[str, Any]]):
        if not articles:
            return

        new_articles = pl.DataFrame({
            "title": [article["title"] for article in articles],
            "description": [article["description"] for article in articles],
            "source": [article["source"]["name"] for article in articles],
            "published_at": pl.Series([article["publishedAt"] for article in articles]).cast(pl.Datetime),
        })

        sentiments = self.sentiment_analyzer(new_articles["title"].to_list())
        entities = [self.extract_entities(title + " " + desc) for title, desc in zip(new_articles["title"], new_articles["description"])]
        
        # Prepare text for topic modeling
        texts = [self.preprocess_text(title + " " + desc) for title, desc in zip(new_articles["title"], new_articles["description"])]
        
        # Update or create LDA model
        self.update_lda_model(texts)
        
        # Get topic distributions
        topic_dists = [self.get_topic_distribution(text) for text in texts]

        new_articles = new_articles.with_columns([
            pl.Series([s["score"] for s in sentiments]).alias("sentiment"),
            pl.Series([self._calculate_relevance(article) for article in articles]).alias("relevance_score"),
            pl.Series(entities).alias("entities"),
            pl.Series(topic_dists).alias("topics")
        ])

        self.news_buffer = pl.concat([self.news_buffer, new_articles]).sort("published_at", descending=True)
        self._trim_buffer()

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        doc = self.nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def preprocess_text(self, text: str) -> List[str]:
        text = re.sub(r'\W+', ' ', text.lower())
        return [token for token in text.split() if token not in STOPWORDS]

    def update_lda_model(self, texts: List[List[str]], num_topics: int = 10):
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(texts)
        
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        if not self.lda_model:
            self.lda_model = LdaMulticore(corpus=corpus, id2word=self.dictionary, num_topics=num_topics)
        else:
            self.lda_model.update(corpus)

    def get_topic_distribution(self, text: List[str]) -> List[Dict[str, float]]:
        bow = self.dictionary.doc2bow(text)
        topic_dist = self.lda_model.get_document_topics(bow)
        return [{"topic_id": topic_id, "score": score} for topic_id, score in topic_dist]

    def _calculate_relevance(self, article: Dict[str, Any]) -> float:
        # Implement a more sophisticated relevance scoring algorithm
        # Consider using entity types and topic distributions
        relevance = 0
        entities = self.extract_entities(article["title"] + " " + article["description"])
        relevant_entity_types = ["ORG", "PERSON", "GPE", "MONEY", "PERCENT"]
        relevance += sum(1 for entity in entities if entity["label"] in relevant_entity_types) / len(relevant_entity_types)

        topic_dist = self.get_topic_distribution(self.preprocess_text(article["title"] + " " + article["description"]))
        relevance += max(topic["score"] for topic in topic_dist) if topic_dist else 0

        return relevance / 2  # Normalize to [0, 1]

    def _trim_buffer(self, max_size: int = 1000):
        if len(self.news_buffer) > max_size:
            self.news_buffer = self.news_buffer.head(max_size)

    def detect_events(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        potential_events = self.news_buffer.filter(
            (pl.col("sentiment").abs() > threshold) & (pl.col("relevance_score") > threshold)
        ).to_dicts()

        return [self._create_event(article) for article in potential_events]

    def _create_event(self, article: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": article["title"],
            "date": article["published_at"],
            "description": article["description"],
            "source": article["source"],
            "sentiment": article["sentiment"],
            "relevance_score": article["relevance_score"],
            "entities": article["entities"],
            "topics": article["topics"]
        }
