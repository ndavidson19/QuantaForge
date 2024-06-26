import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import joblib
import logging

class AdvancedEventImpactPredictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'nn': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        }
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_columns = ['price_impact', 'volatility_impact', 'volume_impact', 'liquidity_impact']

    def prepare_features(self, events: List[Event], market_data: pl.DataFrame) -> pl.DataFrame:
        features = []
        for event in events:
            event_features = {
                'event_type': event.category,
                'event_probability': event.probability,
                'event_duration': event.duration,
                'day_of_week': event.date.weekday(),
                'month': event.date.month,
            }
            
            # Add market context
            market_context = market_data.filter(pl.col('date') == event.date).select(
                ['volatility', 'volume', 'sentiment']
            ).to_dict(as_series=False)
            event_features.update({
                'market_volatility': market_context['volatility'][0],
                'market_volume': market_context['volume'][0],
                'market_sentiment': market_context['sentiment'][0],
            })
            
            # One-hot encode tags
            for tag in event.tags:
                event_features[f'tag_{tag}'] = 1
            
            features.append(event_features)
        
        df = pl.DataFrame(features)
        
        # One-hot encode event_type
        df = df.with_columns([
            pl.when(pl.col('event_type') == category)
              .then(1)
              .otherwise(0)
              .alias(f'event_type_{category}')
            for category in df['event_type'].unique()
        ])
        df = df.drop('event_type')
        
        self.feature_columns = df.columns
        return df

    def prepare_targets(self, events: List[Event]) -> pl.DataFrame:
        targets = []
        for event in events:
            event_targets = {impact.asset: {col: getattr(impact, col) for col in self.target_columns} 
                             for impact in event.impacts}
            targets.append(event_targets)
        return pl.DataFrame(targets)

    def train(self, historical_events: List[Event], market_data: pl.DataFrame, actual_impacts: pl.DataFrame):
        X = self.prepare_features(historical_events, market_data)
        y = actual_impacts

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train.to_numpy())
        X_test_scaled = self.scaler.transform(X_test.to_numpy())

        for asset in y.columns:
            for target in self.target_columns:
                y_train_target = y_train.select(pl.col(f'{asset}.{target}')).to_numpy().flatten()
                y_test_target = y_test.select(pl.col(f'{asset}.{target}')).to_numpy().flatten()

                for name, model in self.models.items():
                    model.fit(X_train_scaled, y_train_target)
                    y_pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test_target, y_pred)
                    logging.info(f"Model {name} MSE for {asset} {target}: {mse}")

        # Train ARIMA model for time series effects
        y_train_values = y_train.to_numpy().flatten()
        self.arima_model = ARIMA(y_train_values, order=(1, 1, 1))
        self.arima_results = self.arima_model.fit()

    def predict_impact(self, event: Event, market_data: pl.DataFrame) -> Dict[str, Dict[str, float]]:
        X = self.prepare_features([event], market_data)
        X_scaled = self.scaler.transform(X.to_numpy())

        predictions = {}
        uncertainties = {}

        for asset in event.impacts[0].asset:  # Assuming all events impact the same assets
            asset_predictions = {}
            asset_uncertainties = {}

            for target in self.target_columns:
                model_predictions = []
                for model in self.models.values():
                    model_predictions.append(model.predict(X_scaled)[0])

                # Ensemble prediction
                mean_prediction = np.mean(model_predictions)
                std_prediction = np.std(model_predictions)

                # Add time series effect
                time_series_effect = self.arima_results.forecast(steps=1)[0]
                final_prediction = mean_prediction + time_series_effect

                # Calculate uncertainty
                uncertainty = norm.interval(0.95, loc=final_prediction, scale=std_prediction)

                asset_predictions[target] = final_prediction
                asset_uncertainties[target] = uncertainty

            predictions[asset] = asset_predictions
            uncertainties[asset] = asset_uncertainties

        return predictions, uncertainties

    def update_models(self, new_events: List[Event], market_data: pl.DataFrame, actual_impacts: pl.DataFrame):
        X_new = self.prepare_features(new_events, market_data)
        y_new = actual_impacts

        X_new_scaled = self.scaler.transform(X_new.to_numpy())

        for asset in y_new.columns:
            for target in self.target_columns:
                y_new_target = y_new.select(pl.col(f'{asset}.{target}')).to_numpy().flatten()

                for model in self.models.values():
                    model.partial_fit(X_new_scaled, y_new_target)

        # Update ARIMA model
        y_new_values = y_new.to_numpy().flatten()
        self.arima_results = self.arima_model.append(y_new_values).fit()

    def save_models(self, path: str):
        joblib.dump(self.models, f"{path}/impact_prediction_models.joblib")
        joblib.dump(self.scaler, f"{path}/impact_prediction_scaler.joblib")
        joblib.dump(self.arima_results, f"{path}/impact_prediction_arima.joblib")

    def load_models(self, path: str):
        self.models = joblib.load(f"{path}/impact_prediction_models.joblib")
        self.scaler = joblib.load(f"{path}/impact_prediction_scaler.joblib")
        self.arima_results = joblib.load(f"{path}/impact_prediction_arima.joblib")

class EventImpactAnalysis:
    def __init__(self, predictor: AdvancedEventImpactPredictor):
        self.predictor = predictor

    def analyze_event_impact(self, event: Event, market_data: pl.DataFrame) -> Dict[str, Any]:
        predictions, uncertainties = self.predictor.predict_impact(event, market_data)
        
        analysis = {
            "event_name": event.name,
            "event_date": event.date,
            "predictions": predictions,
            "uncertainties": uncertainties,
            "interpretation": self._interpret_predictions(predictions, uncertainties)
        }
        
        return analysis

    def _interpret_predictions(self, predictions: Dict[str, Dict[str, float]], 
                               uncertainties: Dict[str, Dict[str, Tuple[float, float]]]) -> str:
        interpretation = ""
        for asset, impacts in predictions.items():
            interpretation += f"For {asset}:\n"
            for impact_type, value in impacts.items():
                uncertainty = uncertainties[asset][impact_type]
                interpretation += f"  - Predicted {impact_type}: {value:.2f} "
                interpretation += f"(95% CI: {uncertainty[0]:.2f} to {uncertainty[1]:.2f})\n"
            interpretation += "\n"
        
        return interpretation

    def simulate_multiple_scenarios(self, event: Event, market_data: pl.DataFrame, num_scenarios: int = 1000) -> pl.DataFrame:
        scenarios = []
        for _ in range(num_scenarios):
            predictions, _ = self.predictor.predict_impact(event, market_data)
            scenarios.append(predictions)
        
        return pl.DataFrame(scenarios)

    def analyze_scenario_distribution(self, scenarios: pl.DataFrame) -> Dict[str, Any]:
        analysis = {}
        for column in scenarios.columns:
            column_stats = scenarios.select(
                pl.col(column).mean().alias('mean'),
                pl.col(column).median().alias('median'),
                pl.col(column).std().alias('std'),
                pl.col(column).quantile(0.025).alias('lower_ci'),
                pl.col(column).quantile(0.975).alias('upper_ci')
            ).to_dict(as_series=False)
            
            analysis[column] = {
                "mean": column_stats['mean'][0],
                "median": column_stats['median'][0],
                "std": column_stats['std'][0],
                "95_ci": (column_stats['lower_ci'][0], column_stats['upper_ci'][0])
            }
        return analysis