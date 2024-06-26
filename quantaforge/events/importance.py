import polars as pl
import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import schedule
import time

class FeatureImportanceModelUpdater:
    def __init__(self, initial_features: List[str], update_frequency: int = 7):
        self.features = initial_features
        self.feature_importances = {feature: 1.0 for feature in initial_features}
        self.models = {
            "price_impact": RandomForestRegressor(n_estimators=100, random_state=42),
            "volatility_impact": RandomForestRegressor(n_estimators=100, random_state=42),
            "volume_impact": RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.update_frequency = update_frequency
        self.last_update = datetime.now()

    def update_feature_importance(self, X: pl.DataFrame, y: pl.DataFrame):
        X_numpy = X.select(self.features).to_numpy()
        X_scaled = self.scaler.fit_transform(X_numpy)

        for impact_type in ["price_impact", "volatility_impact", "volume_impact"]:
            y_impact = y.select(pl.col(impact_type)).to_numpy().ravel()
            importances = mutual_info_regression(X_scaled, y_impact)
            
            for feature, importance in zip(self.features, importances):
                if feature not in self.feature_importances:
                    self.feature_importances[feature] = importance
                else:
                    # Exponential moving average for smooth updates
                    self.feature_importances[feature] = 0.9 * self.feature_importances[feature] + 0.1 * importance

        # Normalize feature importances
        total_importance = sum(self.feature_importances.values())
        self.feature_importances = {f: i / total_importance for f, i in self.feature_importances.items()}

    def update_models(self, X: pl.DataFrame, y: pl.DataFrame):
        X_numpy = X.select(self.features).to_numpy()
        X_scaled = self.scaler.fit_transform(X_numpy)

        # Apply feature importance as sample weights
        sample_weights = np.array([self.feature_importances[f] for f in self.features])
        sample_weights = np.tile(sample_weights, (X_scaled.shape[0], 1))

        for impact_type in ["price_impact", "volatility_impact", "volume_impact"]:
            y_impact = y.select(pl.col(impact_type)).to_numpy().ravel()
            self.models[impact_type].fit(X_scaled, y_impact, sample_weight=sample_weights.sum(axis=1))

    def predict(self, X: pl.DataFrame) -> Dict[str, np.ndarray]:
        X_numpy = X.select(self.features).to_numpy()
        X_scaled = self.scaler.transform(X_numpy)

        predictions = {}
        for impact_type, model in self.models.items():
            predictions[impact_type] = model.predict(X_scaled)

        return predictions

    def get_top_features(self, n: int = 10) -> List[str]:
        return sorted(self.feature_importances, key=self.feature_importances.get, reverse=True)[:n]

    def should_update(self) -> bool:
        return (datetime.now() - self.last_update).days >= self.update_frequency

class EnhancedEventAccuracyEvaluator:
    def __init__(self, initial_features: List[str], lookback_window: int = 30, evaluation_window: int = 5, update_frequency: int = 7):
        self.event_predictions = pl.DataFrame(schema={
            "event_id": pl.Utf8,
            "date": pl.Datetime,
            "asset": pl.Utf8,
            "predicted_price_impact": pl.Float32,
            "predicted_volatility_impact": pl.Float32,
            "predicted_volume_impact": pl.Float32,
            "actual_price_impact": pl.Float32,
            "actual_volatility_impact": pl.Float32,
            "actual_volume_impact": pl.Float32,
            "accuracy_score": pl.Float32
        })
        self.lookback_window = lookback_window
        self.evaluation_window = evaluation_window
        self.model_updater = FeatureImportanceModelUpdater(initial_features, update_frequency)

    def add_event_prediction(self, event: Dict[str, Any]):
        # ... (previous implementation)

    def evaluate_events(self):
        # ... (previous implementation)

        if self.model_updater.should_update():
            self._update_models()

    def _update_models(self):
        recent_events = self.event_predictions.filter(pl.col("accuracy_score").is_not_null())
        
        X = recent_events.select(self.model_updater.features)
        y = recent_events.select(["price_impact", "volatility_impact", "volume_impact"])

        self.model_updater.update_feature_importance(X, y)
        self.model_updater.update_models(X, y)
        self.model_updater.last_update = datetime.now()

        top_features = self.model_updater.get_top_features()
        print(f"Top features after update: {top_features}")

    def predict_impact(self, event_features: Dict[str, Any]) -> Dict[str, float]:
        X = pl.DataFrame([event_features])
        predictions = self.model_updater.predict(X)
        return {
            "price_impact": predictions["price_impact"][0],
            "volatility_impact": predictions["volatility_impact"][0],
            "volume_impact": predictions["volume_impact"][0]
        }

    # ... (other methods remain the same)

class QuantForgeEventAPI:
    def __init__(self, news_api_key: str):
        self.event_detector = EnhancedNLPEventDetector(news_api_key)
        initial_features = [
            "sentiment_score", "relevance_score", "entity_count", "topic_diversity",
            "is_earnings_report", "is_market_crash", "is_policy_change", "is_geopolitical",
            "market_volatility", "trading_volume", "time_of_day", "day_of_week"
        ]
        self.accuracy_evaluator = EnhancedEventAccuracyEvaluator(initial_features)

    def create_event_from_detection(self, event_data: Dict[str, Any]) -> Event:
        event_features = self._extract_event_features(event_data)
        predicted_impacts = self.accuracy_evaluator.predict_impact(event_features)
        
        event = Event(
            id=f"detected_{hash(event_data['name'])}",
            name=event_data["name"],
            date=event_data["date"],
            description=event_data["description"],
            impacts=[EventImpact(asset="SPY", **predicted_impacts)],
            probability=event_data["relevance_score"],
            category=event_data["category"],
            tags=self._generate_tags(event_data)
        )
        
        self.accuracy_evaluator.add_event_prediction(event.__dict__)
        return event

    def _extract_event_features(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sentiment_score": event_data["sentiment"],
            "relevance_score": event_data["relevance_score"],
            "entity_count": len(event_data["entities"]),
            "topic_diversity": len(set(topic["topic_id"] for topic in event_data["topics"])),
            "is_earnings_report": "earnings" in event_data["category"].lower(),
            "is_market_crash": "crash" in event_data["name"].lower() or "crisis" in event_data["name"].lower(),
            "is_policy_change": "policy" in event_data["category"].lower() or "regulation" in event_data["category"].lower(),
            "is_geopolitical": "geopolitical" in event_data["category"].lower(),
            "market_volatility": self._get_market_volatility(),
            "trading_volume": self._get_trading_volume(),
            "time_of_day": event_data["date"].hour,
            "day_of_week": event_data["date"].weekday()
        }

    def _get_market_volatility(self) -> float:
        # Placeholder: Implement logic to get current market volatility
        return 0.5

    def _get_trading_volume(self) -> float:
        # Placeholder: Implement logic to get current trading volume
        return 1000000

    def evaluate_event_accuracy(self):
        self.accuracy_evaluator.evaluate_events()

    def get_accuracy_report(self) -> Dict[str, Any]:
        report = self.accuracy_evaluator.get_accuracy_report()
        report["top_features"] = self.accuracy_evaluator.model_updater.get_top_features()
        return report

    def start_automated_updates(self):
        schedule.every(self.accuracy_evaluator.model_updater.update_frequency).days.do(self.evaluate_event_accuracy)

        while True:
            schedule.run_pending()
            time.sleep(3600)  # Sleep for an hour between checks

# Example usage
if __name__ == "__main__":
    api = QuantForgeEventAPI(news_api_key="your_news_api_key_here")
    
    # Run real-time event detection
    keywords = ["stock market", "economic policy", "central bank", "trade war", "earnings report"]
    asyncio.run(api.monitor_real_time_events(keywords))

    # Start automated updates in a separate thread
    import threading
    update_thread = threading.Thread(target=api.start_automated_updates)
    update_thread.start()

    # Main program continues...
    while True:
        # Periodically check and report on model performance
        accuracy_report = api.get_accuracy_report()
        print("Event Prediction Accuracy Report:")
        print(f"Total events evaluated: {accuracy_report['total_events_evaluated']}")
        print(f"Weighted average accuracy: {accuracy_report['weighted_average_accuracy']:.2f}")
        print(f"Top features: {accuracy_report['top_features']}")
        
        time.sleep(86400)  # Sleep for a day between reports