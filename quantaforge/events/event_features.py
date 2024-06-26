import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import List, Dict

class FeatureImportanceAnalysis:
    def __init__(self, predictor: AdvancedEventImpactPredictor):
        self.predictor = predictor

    def compute_feature_importance(self, X: pl.DataFrame, y: pl.DataFrame, n_repeats: int = 10) -> Dict[str, float]:
        baseline_error = self._compute_baseline_error(X, y)
        importances = {feature: 0 for feature in X.columns}

        for _ in range(n_repeats):
            for feature in X.columns:
                X_permuted = X.with_column(pl.col(feature).shuffle())
                permuted_error = self._compute_baseline_error(X_permuted, y)
                importances[feature] += (permuted_error - baseline_error) / baseline_error

        # Average importances over repeats
        for feature in importances:
            importances[feature] /= n_repeats

        # Normalize importances
        total_importance = sum(importances.values())
        normalized_importances = {feature: importance / total_importance 
                                  for feature, importance in importances.items()}

        return dict(sorted(normalized_importances.items(), key=lambda x: x[1], reverse=True))

    def _compute_baseline_error(self, X: pl.DataFrame, y: pl.DataFrame) -> float:
        predictions = self.predictor.predict_impact_batch(X)
        return mean_squared_error(y.to_numpy().flatten(), predictions.to_numpy().flatten())

    def analyze_feature_importance(self, X: pl.DataFrame, y: pl.DataFrame) -> Dict[str, Any]:
        importances = self.compute_feature_importance(X, y)
        
        analysis = {
            "feature_importances": importances,
            "top_features": list(importances.keys())[:10],
            "interpretation": self._interpret_feature_importance(importances)
        }
        
        return analysis

    def _interpret_feature_importance(self, importances: Dict[str, float]) -> str:
        interpretation = "Feature Importance Analysis:\n\n"
        for feature, importance in list(importances.items())[:10]:
            interpretation += f"{feature}: {importance:.4f}\n"
        
        interpretation += "\nInterpretation:\n"
        interpretation += "- The most important features have the highest values.\n"
        interpretation += "- Features with higher importance have a larger impact on the model's predictions.\n"
        interpretation += "- Consider focusing on the top features for further analysis and model refinement.\n"
        
        return interpretation

    def visualize_feature_importance(self, importances: Dict[str, float], top_n: int = 20):
        import matplotlib.pyplot as plt

        features = list(importances.keys())[:top_n]
        importance_values = [importances[feature] for feature in features]

        plt.figure(figsize=(12, 8))
        plt.bar(features, importance_values)
        plt.title("Top Feature Importances")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

# Add this method to the AdvancedEventImpactPredictor class
def predict_impact_batch(self, X: pl.DataFrame) -> pl.DataFrame:
    X_scaled = self.scaler.transform(X.to_numpy())
    predictions = []
    
    for model in self.models.values():
        model_predictions = model.predict(X_scaled)
        predictions.append(model_predictions)
    
    # Ensemble predictions
    ensemble_predictions = np.mean(predictions, axis=0)
    
    # Add time series effect
    time_series_effects = self.arima_results.forecast(steps=len(X))
    final_predictions = ensemble_predictions + time_series_effects
    
    return pl.DataFrame(final_predictions, schema=self.target_columns)