from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import shap

class ExplainableAI:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain_prediction(self, X):
        shap_values = self.explainer.shap_values(X)
        return dict(zip(self.feature_names, shap_values[0]))

    def plot_feature_importance(self, X):
        shap.summary_plot(self.explainer.shap_values(X), X, feature_names=self.feature_names)


class EnsembleStrategy:
    def __init__(self, strategies, weights=None):
        self.strategies = strategies
        self.weights = weights if weights else [1/len(strategies)] * len(strategies)

    def generate_signal(self, data):
        signals = [strategy.generate_signal(data) for strategy in self.strategies]
        return sum(signal * weight for signal, weight in zip(signals, self.weights))


class MLModule:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def train_model(self, data, target, model_name, model_type='random_forest', **kwargs):
        X = data.drop(columns=[target])
        y = data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(**kwargs)
        # Add more model types as needed
        
        model.fit(X_train_scaled, y_train)
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        return model.score(X_test_scaled, y_test)

    def predict(self, model_name, data):
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        scaled_data = scaler.transform(data)
        return model.predict(scaled_data)

    def save_model(self, model_name, path):
        joblib.dump((self.models[model_name], self.scalers[model_name]), path)

    def load_model(self, model_name, path):
        self.models[model_name], self.scalers[model_name] = joblib.load(path)

    def create_explainer(self, model_name):
        model = self.models[model_name]
        feature_names = self.feature_names[model_name]
        return ExplainableAI(model, feature_names)
    
