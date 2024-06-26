import polars as pl
from typing import List, Dict, Any
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import SGDRegressor
import logging

class EnhancedEventAccuracyEvaluator:
    def __init__(self, lookback_window: int = 30, evaluation_window: int = 5, decay_half_life: int = 30):
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
        self.decay_half_life = decay_half_life
        self.detection_threshold = 0.5  # Initial threshold
        self.adaptive_models = {
            "price": SGDRegressor(loss="huber", alpha=0.01, learning_rate="constant", eta0=0.01),
            "volatility": SGDRegressor(loss="huber", alpha=0.01, learning_rate="constant", eta0=0.01),
            "volume": SGDRegressor(loss="huber", alpha=0.01, learning_rate="constant", eta0=0.01)
        }
        self.alert_threshold = 0.3  # Accuracy threshold for alerts

    def add_event_prediction(self, event: Dict[str, Any]):
        new_prediction = pl.DataFrame([{
            "event_id": event["id"],
            "date": event["date"],
            "asset": impact.asset,
            "predicted_price_impact": impact.price_impact,
            "predicted_volatility_impact": impact.volatility_impact,
            "predicted_volume_impact": impact.volume_impact,
            "actual_price_impact": None,
            "actual_volatility_impact": None,
            "actual_volume_impact": None,
            "accuracy_score": None
        } for impact in event["impacts"]])
        
        self.event_predictions = pl.concat([self.event_predictions, new_prediction])

    def evaluate_events(self):
        cutoff_date = datetime.now() - timedelta(days=self.evaluation_window)
        events_to_evaluate = self.event_predictions.filter(
            (pl.col("date") < cutoff_date) & (pl.col("actual_price_impact").is_null())
        )

        for event in events_to_evaluate.to_dicts():
            self._evaluate_single_event(event)

        self._update_model_weights()
        self._update_detection_threshold()
        self._check_for_alerts()

    def _evaluate_single_event(self, event: Dict[str, Any]):
        asset_data = self._fetch_asset_data(event["asset"], event["date"])
        
        if asset_data is None:
            return

        actual_impacts = self._calculate_actual_impacts(asset_data, event["date"])
        accuracy_score = self._calculate_accuracy_score(event, actual_impacts)

        self.event_predictions = self.event_predictions.with_columns([
            pl.when(pl.col("event_id") == event["event_id"])
              .then(pl.lit(actual_impacts["price_impact"]))
              .otherwise(pl.col("actual_price_impact"))
              .alias("actual_price_impact"),
            pl.when(pl.col("event_id") == event["event_id"])
              .then(pl.lit(actual_impacts["volatility_impact"]))
              .otherwise(pl.col("actual_volatility_impact"))
              .alias("actual_volatility_impact"),
            pl.when(pl.col("event_id") == event["event_id"])
              .then(pl.lit(actual_impacts["volume_impact"]))
              .otherwise(pl.col("actual_volume_impact"))
              .alias("actual_volume_impact"),
            pl.when(pl.col("event_id") == event["event_id"])
              .then(pl.lit(accuracy_score))
              .otherwise(pl.col("accuracy_score"))
              .alias("accuracy_score")
        ])

    def _fetch_asset_data(self, asset: str, event_date: datetime) -> pl.DataFrame:
        # ... (implementation remains the same)

    def _calculate_actual_impacts(self, asset_data: pl.DataFrame, event_date: datetime) -> Dict[str, float]:
        # ... (implementation remains the same)

    def _calculate_accuracy_score(self, event: Dict[str, Any], actual_impacts: Dict[str, float]) -> float:
        # ... (implementation remains the same)

    def _update_model_weights(self):
        recent_events = self.event_predictions.filter(pl.col("accuracy_score").is_not_null())
        
        for impact_type in ["price", "volatility", "volume"]:
            X = recent_events.select([
                f"predicted_{impact_type}_impact",
                "date"
            ]).to_numpy()
            
            y = recent_events.select(f"actual_{impact_type}_impact").to_numpy().ravel()
            
            # Add time-based features
            current_time = datetime.now()
            time_diffs = [(current_time - dt).days for dt in X[:, 1]]
            decay_weights = np.exp(-np.log(2) * np.array(time_diffs) / self.decay_half_life)
            
            X = np.column_stack([X[:, 0], decay_weights])
            
            self.adaptive_models[impact_type].partial_fit(X, y)

    def _update_detection_threshold(self):
        recent_accuracy = self.event_predictions.filter(pl.col("accuracy_score").is_not_null())["accuracy_score"]
        if len(recent_accuracy) > 0:
            mean_accuracy = recent_accuracy.mean()
            std_accuracy = recent_accuracy.std()
            
            # Set threshold to 2 standard deviations below mean accuracy
            self.detection_threshold = max(0.1, mean_accuracy - 2 * std_accuracy)

    def _check_for_alerts(self):
        recent_accuracy = self.event_predictions.filter(pl.col("accuracy_score").is_not_null())["accuracy_score"].mean()
        if recent_accuracy < self.alert_threshold:
            logging.warning(f"Alert: Recent event prediction accuracy ({recent_accuracy:.2f}) is below threshold ({self.alert_threshold})")

    def get_accuracy_report(self) -> Dict[str, Any]:
        evaluated_events = self.event_predictions.filter(pl.col("accuracy_score").is_not_null())
        
        # Apply time-based decay to accuracy scores
        current_time = datetime.now()
        decay_weights = evaluated_events.select(
            ((current_time - pl.col("date")).dt.total_days() * -np.log(2) / self.decay_half_life).exp()
        ).to_numpy().ravel()
        
        weighted_accuracy = (evaluated_events["accuracy_score"] * decay_weights).sum() / decay_weights.sum()
        
        return {
            "total_events_evaluated": len(evaluated_events),
            "weighted_average_accuracy": weighted_accuracy,
            "current_detection_threshold": self.detection_threshold,
            "price_impact_mse": self._weighted_mse(evaluated_events, "price_impact", decay_weights),
            "volatility_impact_mse": self._weighted_mse(evaluated_events, "volatility_impact", decay_weights),
            "volume_impact_mse": self._weighted_mse(evaluated_events, "volume_impact", decay_weights)
        }

    def _weighted_mse(self, df: pl.DataFrame, impact_type: str, weights: np.ndarray) -> float:
        actual = df[f"actual_{impact_type}"].to_numpy()
        predicted = df[f"predicted_{impact_type}"].to_numpy()
        return np.average((actual - predicted) ** 2, weights=weights)

    def predict_impact(self, event_features: Dict[str, Any]) -> Dict[str, float]:
        impacts = {}
        for impact_type in ["price", "volatility", "volume"]:
            model_input = np.array([[
                event_features[f"{impact_type}_impact"],
                1  # Current weight (no decay for new prediction)
            ]])
            impacts[f"{impact_type}_impact"] = self.adaptive_models[impact_type].predict(model_input)[0]
        return impacts

# Update the QuantForgeEventAPI class
class QuantForgeEventAPI:
    def __init__(self, news_api_key: str):
        # ... (previous initializations)
        self.event_detector = EnhancedNLPEventDetector(news_api_key)
        self.accuracy_evaluator = EnhancedEventAccuracyEvaluator()

    # ... (previous methods)

    def create_event_from_detection(self, event_data: Dict[str, Any]) -> Event:
        # Use the adaptive models to predict impacts
        event_features = self._extract_event_features(event_data)
        predicted_impacts = self.accuracy_evaluator.predict_impact(event_features)
        
        # Create the event with predicted impacts
        event = Event(
            id=f"detected_{hash(event_data['name'])}",
            name=event_data["name"],
            date=event_data["date"],
            description=event_data["description"],
            impacts=[EventImpact(asset="SPY", **predicted_impacts)],  # Assuming general market impact
            probability=event_data["relevance_score"],
            category=event_data["category"],
            tags=self._generate_tags(event_data)
        )
        
        # Add the event prediction for future evaluation
        self.accuracy_evaluator.add_event_prediction(event.__dict__)
        
        return event

    def _extract_event_features(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract relevant features from the event data for impact prediction
        # This is a simplified example; you may want to include more sophisticated feature extraction
        return {
            "price_impact": abs(event_data["sentiment"]),
            "volatility_impact": event_data["relevance_score"],
            "volume_impact": len(event_data["entities"])
        }

    def evaluate_event_accuracy(self):
        self.accuracy_evaluator.evaluate_events()

    def get_accuracy_report(self) -> Dict[str, Any]:
        return self.accuracy_evaluator.get_accuracy_report()

# Example usage
if __name__ == "__main__":
    api = QuantForgeEventAPI(news_api_key="your_news_api_key_here")
    
    # Run real-time event detection
    keywords = ["stock market", "economic policy", "central bank", "trade war", "earnings report"]
    asyncio.run(api.monitor_real_time_events(keywords))

    # Evaluate event accuracy (this would typically be run periodically, e.g., daily)
    api.evaluate_event_accuracy()

    # Get accuracy report
    accuracy_report = api.get_accuracy_report()
    print("Event Prediction Accuracy Report:")
    print(f"Total events evaluated: {accuracy_report['total_events_evaluated']}")
    print(f"Weighted average accuracy: {accuracy_report['weighted_average_accuracy']:.2f}")
    print(f"Current detection threshold: {accuracy_report['current_detection_threshold']:.2f}")
    print(f"Price impact MSE: {accuracy_report['price_impact_mse']:.4f}")
    print(f"Volatility impact MSE: {accuracy_report['volatility_impact_mse']:.4f}")
    print(f"Volume impact MSE: {accuracy_report['volume_impact_mse']:.4f}")