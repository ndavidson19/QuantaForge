import polars as pl
from datetime import datetime
from typing import List, Dict, Any

class QuantForgeEventAPI:
    def __init__(self):
        self.event_manager = EventManager()
        self.event_chain_manager = EventChainManager()
        self.impact_predictor = EventImpactPredictor()

    def create_event_from_template(self, template_name: str, **kwargs) -> Event:
        event = getattr(EventTemplate, template_name)(**kwargs)
        self.event_manager.add_event(event)
        return event

    def create_custom_event(self, name: str, date: datetime, description: str, 
                            impacts: List[Dict[str, float]], category: str,
                            tags: List[str], custom_impact_fn: Callable = None) -> Event:
        event = CustomEventCreator.create_custom_event(name, date, description, impacts, category, tags, custom_impact_fn)
        self.event_manager.add_event(event)
        return event

    def create_event_chain(self, chain_id: str) -> EnhancedEventChain:
        return self.event_chain_manager.create_chain(chain_id)

    def add_event_to_chain(self, chain_id: str, event: Event, parent_events: List[Tuple[Event, float]] = None):
        chain = self.event_chain_manager.get_chain(chain_id)
        if chain:
            chain.add_event(event, parent_events)
        else:
            raise ValueError(f"Chain with id {chain_id} not found.")

    def resolve_event_chains(self):
        self.event_chain_manager.resolve_all_chains(self.event_manager)

    def visualize_event_chain(self, chain_id: str):
        self.event_chain_manager.visualize_chain(chain_id)

    # ... (other methods remain the same)

    def simulate_with_events(self, simulation, start_date: datetime, end_date: datetime):
        self.resolve_event_chains()
        events = self.get_events_in_date_range(start_date, end_date)
        self.apply_ml_predictions_to_events(events)
        self.apply_events_to_simulation(simulation, events)
        return simulation.run()
    
    def train_impact_predictor(self, historical_events: List[Event], market_data: pl.DataFrame, actual_impacts: pl.DataFrame):
        self.impact_predictor.train(historical_events, market_data, actual_impacts)

    def predict_event_impact(self, event: Event, market_data: pl.DataFrame) -> Dict[str, Any]:
        return self.impact_analyzer.analyze_event_impact(event, market_data)

    def simulate_event_scenarios(self, event: Event, market_data: pl.DataFrame, num_scenarios: int = 1000) -> Dict[str, Any]:
        scenarios = self.impact_analyzer.simulate_multiple_scenarios(event, market_data, num_scenarios)
        return self.impact_analyzer.analyze_scenario_distribution(scenarios)

    def update_impact_models(self, new_events: List[Event], market_data: pl.DataFrame, actual_impacts: pl.DataFrame):
        self.impact_predictor.update_models(new_events, market_data, actual_impacts)

    def save_impact_models(self, path: str):
        self.impact_predictor.save_models(path)

    def load_impact_models(self, path: str):
        self.impact_predictor.load_models(path)

    def simulate_with_events(self, simulation, start_date: datetime, end_date: datetime, market_data: pl.DataFrame):
        self.resolve_event_chains()
        events = self.get_events_in_date_range(start_date, end_date)
        
        for event in events:
            event_market_data = market_data.filter(pl.col('date') <= event.date)
            impact_analysis = self.predict_event_impact(event, event_market_data)
            event.predicted_impacts = impact_analysis['predictions']
            event.impact_uncertainties = impact_analysis['uncertainties']
        
        self.apply_events_to_simulation(simulation, events)
        return simulation.run()

    def apply_events_to_simulation(self, simulation, events: List[Event]):
        for event in events:
            for asset, impacts in event.predicted_impacts.items():
                simulation.apply_market_impact(
                    asset=asset,
                    price_impact=impacts['price_impact'],
                    volatility_impact=impacts['volatility_impact'],
                    volume_impact=impacts['volume_impact'],
                    liquidity_impact=impacts['liquidity_impact'],
                    date=event.date,
                    duration=event.duration,
                    uncertainty=event.impact_uncertainties[asset]
                )

    def analyze_feature_importance(self, historical_events: List[Event], market_data: pl.DataFrame, actual_impacts: pl.DataFrame):
        X = self.impact_predictor.prepare_features(historical_events, market_data)
        y = actual_impacts
        return self.feature_importance_analyzer.analyze_feature_importance(X, y)

    def visualize_feature_importance(self, importances: Dict[str, float], top_n: int = 20):
        self.feature_importance_analyzer.visualize_feature_importance(importances, top_n)

    def fine_tune_ner(self, training_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]], n_iter: int = 10):
        self.event_detector.fine_tune_ner(training_data, n_iter)

    def detect_trends(self, window_size: int = 100) -> List[Dict[str, Any]]:
        return self.event_detector.detect_trends(window_size)

    def create_event_from_detection(self, event_data: Dict[str, Any]) -> Event:
        impact = self._calculate_impact(event_data)
        linked_entities = [entity for entity in event_data["entities"] if entity["kb_id"]]

        return Event(
            id=f"detected_{hash(event_data['name'])}",
            name=event_data["name"],
            date=event_data["date"],
            description=event_data["description"],
            impacts=[impact],
            probability=event_data["relevance_score"],
            category=event_data["category"],
            tags=self._generate_tags(event_data),
            linked_entities=linked_entities
        )
    

    def _calculate_impact(self, event_data: Dict[str, Any]) -> EventImpact:
        # Implement more sophisticated impact calculation based on NLP results
        sentiment = event_data["sentiment"]
        relevance = event_data["relevance_score"]
        entity_importance = self._calculate_entity_importance(event_data["entities"])
        topic_importance = self._calculate_topic_importance(event_data["topics"])

        price_impact = sentiment * relevance * (entity_importance + topic_importance) / 2
        volatility_impact = abs(sentiment) * relevance * (entity_importance + topic_importance) / 2
        volume_impact = relevance * (entity_importance + topic_importance) / 2

        return EventImpact(
            asset="SPY",  # Assuming a general market impact
            price_impact=price_impact,
            volatility_impact=volatility_impact,
            volume_impact=volume_impact
        )

    def _calculate_entity_importance(self, entities: List[Dict[str, str]]) -> float:
        # Implement logic to calculate importance based on entity types and frequencies
        important_types = {"ORG": 0.5, "PERSON": 0.3, "GPE": 0.4, "MONEY": 0.6, "PERCENT": 0.4}
        return sum(important_types.get(entity["label"], 0.1) for entity in entities) / len(entities) if entities else 0

    def _calculate_topic_importance(self, topics: List[Dict[str, float]]) -> float:
        # Implement logic to calculate importance based on topic scores
        return max(topic["score"] for topic in topics) if topics else 0

    def _determine_category(self, event_data: Dict[str, Any]) -> str:
        # Implement logic to determine event category based on entities and topics
        entities = [entity["label"] for entity in event_data["entities"]]
        if "ORG" in entities and "MONEY" in entities:
            return "corporate_finance"
        elif "GPE" in entities and "PERSON" in entities:
            return "geopolitical"
        elif "PERCENT" in entities and "MONEY" in entities:
            return "economic_indicator"
        else:
            return "general"

    def _generate_tags(self, event_data: Dict[str, Any]) -> List[str]:
        # Generate tags based on entities, topics, and source
        tags = ["real-time", event_data["source"]]
        tags.extend([entity["text"] for entity in event_data["entities"][:5]])  # Add top 5 entities as tags
        tags.extend([f"topic_{topic['topic_id']}" for topic in event_data["topics"][:3]])  # Add top 3 topics as tags
        return list(set(tags))  # Remove duplicates


    def create_event_from_detection(self, event_data: Dict[str, Any]) -> Event:
        event = super().create_event_from_detection(event_data)
        self.accuracy_evaluator.add_event_prediction(event.__dict__)
        return event

    def evaluate_event_accuracy(self):
        self.accuracy_evaluator.evaluate_events()

    def get_accuracy_report(self) -> Dict[str, Any]:
        return self.accuracy_evaluator.get_accuracy_report()