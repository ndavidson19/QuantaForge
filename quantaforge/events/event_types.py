from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Callable

@dataclass
class EventImpact:
    asset: str
    price_impact: float
    volatility_impact: float
    volume_impact: float = 0.0
    liquidity_impact: float = 0.0

@dataclass
class Event:
    id: str
    name: str
    date: datetime
    description: str
    impacts: List[EventImpact]
    probability: float = 1.0
    duration: int = 1
    category: str = "custom"
    tags: List[str] = field(default_factory=list)
    custom_impact_fn: Callable = None

class EventTemplate:
    @staticmethod
    def stock_split(company: str, date: datetime, split_ratio: float) -> Event:
        price_impact = (1 / split_ratio) - 1
        volume_impact = split_ratio - 1
        
        impact = EventImpact(
            asset=company,
            price_impact=price_impact,
            volatility_impact=0.1,  # Assuming a small volatility increase
            volume_impact=volume_impact
        )
        
        return Event(
            id=f"{company}_split_{date}",
            name=f"{company} Stock Split",
            date=date,
            description=f"{company} announces a {split_ratio}-for-1 stock split",
            impacts=[impact],
            category="corporate_action",
            tags=[company, "stock_split"]
        )

    @staticmethod
    def merger_announcement(acquirer: str, target: str, date: datetime, premium: float) -> Event:
        acquirer_impact = EventImpact(
            asset=acquirer,
            price_impact=-0.02,  # Assuming a small negative impact on acquirer
            volatility_impact=0.2,
            volume_impact=0.5
        )
        
        target_impact = EventImpact(
            asset=target,
            price_impact=premium,
            volatility_impact=0.3,
            volume_impact=1.0
        )
        
        return Event(
            id=f"{acquirer}_{target}_merger_{date}",
            name=f"{acquirer} to Acquire {target}",
            date=date,
            description=f"{acquirer} announces plans to acquire {target} at a {premium:.1%} premium",
            impacts=[acquirer_impact, target_impact],
            category="corporate_action",
            tags=[acquirer, target, "merger"]
        )

    @staticmethod
    def product_launch(company: str, date: datetime, impact_estimate: float) -> Event:
        impact = EventImpact(
            asset=company,
            price_impact=impact_estimate,
            volatility_impact=0.2,
            volume_impact=0.5
        )
        
        return Event(
            id=f"{company}_product_launch_{date}",
            name=f"{company} Product Launch",
            date=date,
            description=f"{company} launches a new product",
            impacts=[impact],
            category="corporate_event",
            tags=[company, "product_launch"]
        )

class CustomEventCreator:
    @staticmethod
    def create_custom_event(name: str, date: datetime, description: str, 
                            impacts: List[Dict[str, float]], category: str,
                            tags: List[str], custom_impact_fn: Callable = None) -> Event:
        event_impacts = [EventImpact(**impact) for impact in impacts]
        return Event(
            id=f"custom_{name.lower().replace(' ', '_')}_{date}",
            name=name,
            date=date,
            description=description,
            impacts=event_impacts,
            category=category,
            tags=tags,
            custom_impact_fn=custom_impact_fn
        )