import polars as pl
from typing import List, Dict, Any
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.metrics import mean_squared_error
import numpy as np

class EventAccuracyEvaluator:
    def __init__(self, lookback_window: int = 30, evaluation_window: int = 5):
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
        start_date = event_date - timedelta(days=self.lookback_window)
        end_date = event_date + timedelta(days=self.evaluation_window)
        
        try:
            data = yf.download(asset, start=start_date, end=end_date)
            return pl.DataFrame(data).with_columns(pl.col("Date").cast(pl.Datetime))
        except Exception as e:
            print(f"Error fetching data for {asset}: {e}")
            return None

    def _calculate_actual_impacts(self, asset_data: pl.DataFrame, event_date: datetime) -> Dict[str, float]:
        pre_event = asset_data.filter(pl.col("Date") < event_date).tail(self.lookback_window)
        post_event = asset_data.filter(pl.col("Date") >= event_date).head(self.evaluation_window)

        pre_price = pre_event["Close"].mean()
        post_price = post_event["Close"].mean()
        price_impact = (post_price - pre_price) / pre_price

        pre_volatility = pre_event["Close"].std()
        post_volatility = post_event["Close"].std()
        volatility_impact = (post_volatility - pre_volatility) / pre_volatility

        pre_volume = pre_event["Volume"].mean()
        post_volume = post_event["Volume"].mean()
        volume_impact = (post_volume - pre_volume) / pre_volume

        return {
            "price_impact": price_impact,
            "volatility_impact": volatility_impact,
            "volume_impact": volume_impact
        }

    def _calculate_accuracy_score(self, event: Dict[str, Any], actual_impacts: Dict[str, float]) -> float:
        predicted = np.array([
            event["predicted_price_impact"],
            event["predicted_volatility_impact"],
            event["predicted_volume_impact"]
        ])
        actual = np.array([
            actual_impacts["price_impact"],
            actual_impacts["volatility_impact"],
            actual_impacts["volume_impact"]
        ])
        
        mse = mean_squared_error(actual, predicted)
        return 1 / (1 + mse)  # Convert MSE to a score between 0 and 1

    def _update_model_weights(self):
        # This is a placeholder for updating the event impact prediction model
        # In a real implementation, you would use the accuracy scores to adjust the model
        recent_accuracy = self.event_predictions.filter(pl.col("accuracy_score").is_not_null())
        avg_accuracy = recent_accuracy["accuracy_score"].mean()
        print(f"Average recent prediction accuracy: {avg_accuracy:.2f}")

    def get_accuracy_report(self) -> Dict[str, Any]:
        evaluated_events = self.event_predictions.filter(pl.col("accuracy_score").is_not_null())
        
        return {
            "total_events_evaluated": len(evaluated_events),
            "average_accuracy": evaluated_events["accuracy_score"].mean(),
            "price_impact_mse": mean_squared_error(
                evaluated_events["actual_price_impact"],
                evaluated_events["predicted_price_impact"]
            ),
            "volatility_impact_mse": mean_squared_error(
                evaluated_events["actual_volatility_impact"],
                evaluated_events["predicted_volatility_impact"]
            ),
            "volume_impact_mse": mean_squared_error(
                evaluated_events["actual_volume_impact"],
                evaluated_events["predicted_volume_impact"]
            )
        }