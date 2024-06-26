import polars as pl
from typing import List, Dict, Any, Tuple
import aiohttp
import asyncio
from transformers import pipeline
import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import yfinance as yf

class EnhancedNLPEventDetector:
    def __init__(self, api_key: str, sentiment_model: str = "finiteautomata/bertweet-base-sentiment-analysis"):
        self.api_key = api_key
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        self.nlp = spacy.load("en_core_web_sm")
        self.lda_model = None
        self.dictionary = None
        self.entity_kb = self._load_entity_kb()
        self.event_classifier = self._train_event_classifier()
        self.news_buffer = pl.DataFrame(schema={
            "title": pl.Utf8,
            "description": pl.Utf8,
            "source": pl.Utf8,
            "published_at": pl.Datetime,
            "sentiment": pl.Float32,
            "relevance_score": pl.Float32,
            "entities": pl.List(pl.Struct({"text": pl.Utf8, "label": pl.Utf8, "kb_id": pl.Utf8})),
            "topics": pl.List(pl.Struct({"topic_id": pl.UInt32, "score": pl.Float32})),
            "category": pl.Utf8
        })

    def _load_entity_kb(self) -> Dict[str, str]:
        # Load a knowledge base of entities (e.g., companies, people, locations)
        # This is a placeholder - in a real implementation, you'd load from a database
        return {
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Amazon": "AMZN",
            "Federal Reserve": "FED",
            "European Union": "EU"
        }

    def _train_event_classifier(self) -> RandomForestClassifier:
        # Train a classifier to categorize events
        # This is a placeholder - in a real implementation, you'd use a larger, labeled dataset
        data = [
            ("Apple announces new iPhone", "product_launch"),
            ("Federal Reserve raises interest rates", "economic_policy"),
            ("Amazon acquires Whole Foods", "merger_acquisition"),
            ("Trade tensions escalate between US and China", "geopolitical"),
            ("Microsoft reports strong quarterly earnings", "earnings_report")
        ]
        X, y = zip(*data)
        X = self.nlp.pipe(X)
        X_vectorized = [doc.vector for doc in X]
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_vectorized, y)
        return clf

    async def fetch_news(self, keywords: List[str]):
        # Existing implementation...

    def fine_tune_ner(self, training_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]], n_iter: int = 10):
        # Convert training data to spaCy format
        train_data = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            ents = []
            for start, end, label in annotations.get("entities", []):
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    ents.append(span)
            doc.ents = ents
            train_data.append(doc)

        # Create a new NER pipe
        ner = self.nlp.create_pipe("ner")
        self.nlp.add_pipe(ner, last=True)

        # Add new entity labels
        for _, annotations in training_data:
            for _, _, label in annotations.get("entities", []):
                ner.add_label(label)

        # Fine-tune the model
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            for itn in range(n_iter):
                losses = {}
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    self.nlp.update(batch, drop=0.5, sgd=optimizer, losses=losses)
                print(f"Losses at iteration {itn}: {losses}")

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
        entities = [self.extract_and_link_entities(title + " " + desc) for title, desc in zip(new_articles["title"], new_articles["description"])]
        
        texts = [self.preprocess_text(title + " " + desc) for title, desc in zip(new_articles["title"], new_articles["description"])]
        self.update_lda_model(texts)
        topic_dists = [self.get_topic_distribution(text) for text in texts]

        categories = self.classify_events(new_articles["title"].to_list())

        new_articles = new_articles.with_columns([
            pl.Series([s["score"] for s in sentiments]).alias("sentiment"),
            pl.Series([self._calculate_relevance(article) for article in articles]).alias("relevance_score"),
            pl.Series(entities).alias("entities"),
            pl.Series(topic_dists).alias("topics"),
            pl.Series(categories).alias("category")
        ])

        self.news_buffer = pl.concat([self.news_buffer, new_articles]).sort("published_at", descending=True)
        self._trim_buffer()

    def extract_and_link_entities(self, text: str) -> List[Dict[str, str]]:
        doc = self.nlp(text)
        linked_entities = []
        for ent in doc.ents:
            linked_entity = {"text": ent.text, "label": ent.label_, "kb_id": ""}
            if ent.text in self.entity_kb:
                linked_entity["kb_id"] = self.entity_kb[ent.text]
            linked_entities.append(linked_entity)
        return linked_entities

    def classify_events(self, titles: List[str]) -> List[str]:
        docs = self.nlp.pipe(titles)
        X_vectorized = [doc.vector for doc in docs]
        return self.event_classifier.predict(X_vectorized)

    def detect_trends(self, window_size: int = 100) -> List[Dict[str, Any]]:
        if len(self.news_buffer) < window_size:
            return []

        recent_news = self.news_buffer.head(window_size)
        topic_trends = self._calculate_topic_trends(recent_news)
        entity_trends = self._calculate_entity_trends(recent_news)

        return [
            {"type": "topic", "id": trend["id"], "change": trend["change"]}
            for trend in topic_trends
        ] + [
            {"type": "entity", "id": trend["id"], "change": trend["change"]}
            for trend in entity_trends
        ]

    def _calculate_topic_trends(self, news: pl.DataFrame) -> List[Dict[str, Any]]:
        topic_counts = news["topics"].explode().to_struct().groupby("topic_id").agg(pl.col("score").sum())
        topic_counts = topic_counts.sort("score", descending=True)
        
        prev_counts = topic_counts.head(len(topic_counts) // 2)
        curr_counts = topic_counts.tail(len(topic_counts) // 2)
        
        trends = []
        for topic_id in topic_counts["topic_id"]:
            prev_score = prev_counts.filter(pl.col("topic_id") == topic_id)["score"].sum()
            curr_score = curr_counts.filter(pl.col("topic_id") == topic_id)["score"].sum()
            change = (curr_score - prev_score) / prev_score if prev_score > 0 else float('inf')
            trends.append({"id": topic_id, "change": change})
        
        return sorted(trends, key=lambda x: abs(x["change"]), reverse=True)

    def _calculate_entity_trends(self, news: pl.DataFrame) -> List[Dict[str, Any]]:
        entities = news["entities"].explode().to_struct()
        entity_counts = entities.groupby("text").agg(pl.count())
        entity_counts = entity_counts.sort("count", descending=True)
        
        prev_counts = entity_counts.head(len(entity_counts) // 2)
        curr_counts = entity_counts.tail(len(entity_counts) // 2)
        
        trends = []
        for entity in entity_counts["text"]:
            prev_count = prev_counts.filter(pl.col("text") == entity)["count"].sum()
            curr_count = curr_counts.filter(pl.col("text") == entity)["count"].sum()
            change = (curr_count - prev_count) / prev_count if prev_count > 0 else float('inf')
            trends.append({"id": entity, "change": change})
        
        return sorted(trends, key=lambda x: abs(x["change"]), reverse=True)