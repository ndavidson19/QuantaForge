import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timedelta
from core import Event, EventManager, EventImpact
from templates import EventTemplate
from chaining import EventChain, ChainedEventGenerator
from ml_integration import EventImpactPredictor, EventDataPreparation

class QuantForgeEventAPI:
    def __init__(self):
        self.event_manager = EventManager()
        self.chained_event_generator = ChainedEventGenerator(self.event_manager)
        self.impact_predictor = EventImpactPredictor()

    def create_event(self, event_data: Dict) -> Event:
        event = Event(**event_data)
        self.event_manager.add_event(event)
        return event

    def create_event_from_template(self, template_name: str, **kwargs) -> Event:
        if template_name == "earnings_announcement":
            event = EventTemplate.earnings_announcement(**kwargs)
        elif template_name == "fed_rate_decision":
            event = EventTemplate.fed_rate_decision(**kwargs)
        elif template_name == "geopolitical_event":
            event = EventTemplate.geopolitical_event(**kwargs)
        else:
            raise ValueError(f"Unknown template: {template_name}")
        
        self.event_manager.add_event(event)
        return event

    def create_event_chain(self, events: List[Dict], probabilities: List[float]) -> EventChain:
        if len(events) != len(probabilities):
            raise ValueError("Number of events must match number of probabilities")
        
        chain = EventChain(Event(**events[0]))
        for event_data, prob in zip(events[1:], probabilities[1:]):
            chain.add_event(Event(**event_data), prob)
        
        self.chained_event_generator.add_chain(chain)
        return chain

    def generate_chained_events(self):
        self.chained_event_generator.generate_events()

    def predict_event_impact(self, event: Event) -> List[EventImpact]:
        return self.impact_predictor.predict_impact(event)

    def train_impact_predictor(self, historical_events: List[Event], market_data: pd.DataFrame):
        prepared_events, actual_impacts = EventDataPreparation.prepare_historical_data(historical_events, market_data)
        self.impact_predictor.train(prepared_events, actual_impacts)

    def get_events_for_date(self, date: datetime) -> List[Event]:
        return self.event_manager.get_events_for_date(date)

    def get_events_by_category(self, category: str) -> List[Event]:
        return self.event_manager.get_events_by_category(category)

    def get_events_by_tag(self, tag: str) -> List[Event]:
        return self.event_manager.get_events_by_tag(tag)

    def apply_events_to_simulation(self, simulation, events: List[Event]):
        for event in events:
            for impact in event.impacts:
                simulation.apply_market_impact(
                    asset=impact.asset,
                    price_impact=impact.price_impact,
                    volatility_impact=impact.volatility_impact,
                    volume_impact=impact.volume_impact,
                    date=event.date,
                    duration=event.duration
                )

    def create_custom_event(self, name: str, date: datetime, description: str, 
                            impacts: List[Dict[str, float]], probability: float = 1.0, 
                            duration: int = 1, category: str = "custom", tags: List[str] = None) -> Event:
        event_impacts = [EventImpact(**impact) for impact in impacts]
        event = Event(name=name, date=date, description=description, impacts=event_impacts,
                      probability=probability, duration=duration, category=category, tags=tags or [])
        self.event_manager.add_event(event)
        return event

    def create_market_crash_chain(self, start_date: datetime, initial_drop: float):
        chain = self.chained_event_generator.create_market_crash_chain(start_date, initial_drop)
        return chain

    def get_all_events(self) -> List[Event]:
        return list(self.event_manager.events.values())

    def remove_event(self, event_id: str):
        self.event_manager.remove_event(event_id)

    def update_event(self, event_id: str, **kwargs):
        event = self.event_manager.get_event(event_id)
        if event:
            for key, value in kwargs.items():
                setattr(event, key, value)
        else:
            raise ValueError(f"Event with id {event_id} not found")

    def get_events_in_date_range(self, start_date: datetime, end_date: datetime) -> List[Event]:
        return [event for event in self.event_manager.events.values() 
                if start_date <= event.date <= end_date]

    def apply_ml_predictions_to_events(self, events: List[Event]):
        for event in events:
            predicted_impacts = self.predict_event_impact(event)
            event.impacts = predicted_impacts

    def simulate_with_events(self, simulation, start_date: datetime, end_date: datetime):
        events = self.get_events_in_date_range(start_date, end_date)
        self.apply_ml_predictions_to_events(events)
        self.apply_events_to_simulation(simulation, events)
        return simulation.run()

    def export_events_to_csv(self, file_path: str):
        events_data = []
        for event in self.get_all_events():
            for impact in event.impacts:
                events_data.append({
                    'id': event.id,
                    'name': event.name,
                    'date': event.date,
                    'description': event.description,
                    'probability': event.probability,
                    'duration': event.duration,
                    'category': event.category,
                    'tags': ','.join(event.tags),
                    'asset': impact.asset,
                    'price_impact': impact.price_impact,
                    'volatility_impact': impact.volatility_impact,
                    'volume_impact': impact.volume_impact
                })
        df = pd.DataFrame(events_data)
        df.to_csv(file_path, index=False)

    def import_events_from_csv(self, file_path: str):
        df = pd.read_csv(file_path, parse_dates=['date'])
        events = {}
        for _, row in df.iterrows():
            event_id = row['id']
            if event_id not in events:
                events[event_id] = Event(
                    id=event_id,
                    name=row['name'],
                    date=row['date'],
                    description=row['description'],
                    impacts=[],
                    probability=row['probability'],
                    duration=row['duration'],
                    category=row['category'],
                    tags=row['tags'].split(',') if pd.notna(row['tags']) else []
                )
            events[event_id].impacts.append(EventImpact(
                asset=row['asset'],
                price_impact=row['price_impact'],
                volatility_impact=row['volatility_impact'],
                volume_impact=row['volume_impact']
            ))
        for event in events.values():
            self.event_manager.add_event(event)



