from quantforge import Strategy, Indicator, Signal
from quantforge.events import Event, EventImpact

class EarningsMomentumStrategy(Strategy):
    def __init__(self, symbol, lookback_period=20, momentum_threshold=0.05, 
                 sentiment_threshold=0.7, stop_loss=0.05):
        super().__init__("Earnings Momentum Strategy")
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.sentiment_threshold = sentiment_threshold
        self.stop_loss = stop_loss
        
        # Initialize indicators
        self.add_indicator(Indicator("RSI", {"window": 14}, name="RSI"))
        self.add_indicator(Indicator("SMA", {"window": 50}, name="SMA_50"))
        self.add_indicator(Indicator("SMA", {"window": 200}, name="SMA_200"))
        
        # Initialize custom sentiment indicator
        self.add_indicator(Indicator("CustomSentiment", {}, name="Sentiment"))

    def calculate_momentum(self, data):
        return (data['close'][-1] / data['close'][-self.lookback_period]) - 1

    def generate_signals(self, data):
        momentum = self.calculate_momentum(data)
        rsi = data['RSI'][-1]
        sentiment = data['Sentiment'][-1]
        sma_50 = data['SMA_50'][-1]
        sma_200 = data['SMA_200'][-1]
        current_price = data['close'][-1]

        # Check if we're in an uptrend
        uptrend = sma_50 > sma_200 and current_price > sma_50

        # Generate buy signal
        if (momentum > self.momentum_threshold and 
            rsi > 50 and 
            sentiment > self.sentiment_threshold and 
            uptrend and 
            not self.position):
            return Signal.BUY

        # Generate sell signal
        elif (self.position and 
              (sentiment < self.sentiment_threshold or 
               current_price < self.position.entry_price * (1 - self.stop_loss))):
            return Signal.SELL

        return Signal.HOLD

    def on_event(self, event: Event):
        if event.category == 'earnings':
            # If it's an earnings event, check the sentiment and impact
            sentiment = event.impacts[0].sentiment if event.impacts else 0
            price_impact = event.impacts[0].price_impact if event.impacts else 0
            
            if sentiment < 0 or price_impact < 0:
                # If sentiment or predicted impact is negative, exit the position
                return Signal.SELL
        
        return Signal.HOLD
    

class CustomSentimentIndicator(Indicator):
    def __init__(self, params):
        super().__init__("CustomSentiment", params)
        self.sentiment_api = SentimentAPI()  # Assume we have a sentiment API

    def calculate(self, data):
        # Fetch sentiment data for the last trading day
        sentiment_data = self.sentiment_api.get_sentiment(data.index[-1], data['symbol'][-1])
        
        # Aggregate sentiment scores (e.g., from news, social media, analyst ratings)
        sentiment_score = np.mean([
            sentiment_data['news_sentiment'],
            sentiment_data['social_media_sentiment'],
            sentiment_data['analyst_sentiment']
        ])
        
        return sentiment_score
    

class EarningsEventTemplate(EventTemplate):
    @staticmethod
    def earnings_announcement(company: str, date: datetime, eps_surprise: float, sentiment: float):
        impact = EventImpact(
            asset=company,
            price_impact=eps_surprise * 0.05,  # Simplified impact calculation
            volatility_impact=abs(eps_surprise) * 0.1,
            sentiment=sentiment
        )
        return Event(
            name=f"{company} Earnings Announcement",
            date=date,
            description=f"{company} announces earnings with EPS surprise of {eps_surprise}",
            impacts=[impact],
            category="earnings",
            tags=[company, "earnings"]
        )