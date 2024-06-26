
class MacroIndicator(Indicator):
    def __init__(self, params):
        super().__init__("CustomMacro", params)
        self.macro_data_api = MacroDataAPI()  # Assume we have a macro data API

    def calculate(self, data):
        # Fetch latest macro data
        macro_data = self.macro_data_api.get_latest_data()
        
        # Define sector preferences based on macro conditions
        sector_scores = {
            'XLK': 0,  # Technology
            'XLF': 0,  # Financials
            'XLE': 0,  # Energy
            'XLV': 0,  # Healthcare
            'XLY': 0,  # Consumer Discretionary
            'XLP': 0,  # Consumer Staples
            'XLI': 0,  # Industrials
            'XLB': 0,  # Materials
            'XLRE': 0, # Real Estate
            'XLU': 0   # Utilities
        }

        # Score sectors based on macro conditions
        if macro_data['gdp_growth'] > 2.5:
            sector_scores['XLY'] += 1  # Consumer Discretionary tends to do well in strong growth
            sector_scores['XLF'] += 1  # Financials benefit from growth
        elif macro_data['gdp_growth'] < 1:
            sector_scores['XLP'] += 1  # Consumer Staples are defensive
            sector_scores['XLU'] += 1  # Utilities are defensive

        if macro_data['inflation'] > 3:
            sector_scores['XLE'] += 1  # Energy often benefits from inflation
            sector_scores['XLB'] += 1  # Materials can benefit from inflation
        elif macro_data['inflation'] < 1:
            sector_scores['XLK'] += 1  # Technology might benefit in low inflation

        if macro_data['unemployment'] > 6:
            sector_scores['XLP'] += 1  # Consumer Staples are defensive
            sector_scores['XLV'] += 1  # Healthcare is relatively inelastic
        elif macro_data['unemployment'] < 4:
            sector_scores['XLF'] += 1  # Financials benefit from strong employment
            sector_scores['XLY'] += 1  # Consumer Discretionary benefits from strong employment

        # Normalize scores
        max_score = max(sector_scores.values())
        if max_score > 0:
            sector_scores = {k: v / max_score for k, v in sector_scores.items()}

        return pd.Series(sector_scores)
    


class EconomicEventTemplate(EventTemplate):
    @staticmethod
    def economic_release(date: datetime, indicator: str, value: float, expectation: float):
        surprise = (value - expectation) / expectation
        impact = EventImpact(
            asset="SPY",  # Use S&P 500 as a proxy for overall market impact
            price_impact=surprise * 0.01,  # Simplified impact calculation
            volatility_impact=abs(surprise) * 0.05
        )
        return Event(
            name=f"{indicator} Economic Release",
            date=date,
            description=f"{indicator} released with value {value}, vs expectation {expectation}",
            impacts=[impact],
            category="economic",
            tags=[indicator, "economic"]
        )