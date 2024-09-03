import polars as pl
import numpy as np
from typing import List, Dict, Any
from quantaforge.strategy import Strategy
from quantaforge.indicators import SMA, ATR, RSI
from quantaforge.signals import CrossOverSignal, ThresholdSignal
from quantaforge.condition import Condition
from quantaforge.order import Order, OrderType
from quantaforge.generics import Portfolio

class MarketRiskProtectionStrategy(Strategy):
    def __init__(self, 
                 symbols: List[str] = ['SPY'],
                 sma_window: int = 200,
                 atr_window: int = 14,
                 rsi_window: int = 14,
                 volatility_threshold: float = 0.2,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 put_option_strike_pct: float = 0.9,
                 put_option_expiry_days: int = 30):
        super().__init__("Market Risk Protection Strategy")
        self.symbols = symbols
        self.sma_window = sma_window
        self.atr_window = atr_window
        self.rsi_window = rsi_window
        self.volatility_threshold = volatility_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.put_option_strike_pct = put_option_strike_pct
        self.put_option_expiry_days = put_option_expiry_days

        # Add indicators
        self.add_indicator(f"SMA_{sma_window}", SMA(window=sma_window))
        self.add_indicator(f"ATR_{atr_window}", ATR(window=atr_window))
        self.add_indicator(f"RSI_{rsi_window}", RSI(window=rsi_window))

        # Add signals
        self.add_signal("SMA_Crossover", CrossOverSignal("close", f"SMA_{sma_window}", name="SMA_Crossover"))
        self.add_signal("RSI_Oversold", ThresholdSignal(f"RSI_{rsi_window}", self.rsi_oversold, 100, name="RSI_Oversold"))
        self.add_signal("RSI_Overbought", ThresholdSignal(f"RSI_{rsi_window}", 0, self.rsi_overbought, name="RSI_Overbought"))

        # Add entry conditions
        self.add_entry_condition(Condition("SMA_Crossover_buy", "==", 1))
        self.add_entry_condition(Condition("RSI_Oversold_buy", "==", 1))

        # Add exit conditions
        self.add_exit_condition(Condition("SMA_Crossover_sell", "==", 1))
        self.add_exit_condition(Condition("RSI_Overbought_sell", "==", 1))

    def calculate_position_size(self, portfolio_value: float, current_price: float, atr: float) -> int:
        volatility = atr / current_price
        if volatility > self.volatility_threshold:
            return 0
        else:
            position_size = (portfolio_value * 0.01) / (atr * 2)  # Risk 1% of portfolio per trade
            return int(position_size)

    def generate_orders(self, data: Dict[str, Any], portfolio: Portfolio) -> List[Order]:
        orders = []
        symbol = data['symbol']
        current_position = portfolio.positions.get(symbol, None)
        
        # Check entry conditions
        entry_triggered, entry_reasons = self._check_entry_conditions(data)
        if entry_triggered:
            desired_position_size = self.calculate_position_size(portfolio.total_value, data['close'], data[f'ATR_{self.atr_window}'])
            if current_position:
                # Adjust existing position
                quantity_difference = desired_position_size - current_position.quantity
                if quantity_difference != 0:
                    orders.append(Order(symbol, quantity_difference, OrderType.MARKET, data['close'], data['timestamp']))
            else:
                # Open new position
                orders.append(Order(symbol, desired_position_size, OrderType.MARKET, data['close'], data['timestamp']))
        
        # Check exit conditions
        exit_triggered, exit_reasons = self._check_exit_conditions(data, symbol)
        if exit_triggered and current_position:
            # Close existing position
            orders.append(Order(symbol, -current_position.quantity, OrderType.MARKET, data['close'], data['timestamp']))
        
        # Add protective put orders
        orders.extend(self._place_protective_put(data, portfolio))
        
        return orders

    def _place_protective_put(self, data: Dict[str, Any], portfolio: Portfolio) -> List[Order]:
        orders = []
        symbol = data['symbol']
        current_position = portfolio.positions.get(symbol, None)
        
        if current_position and data['put_option_signal']:
            put_option_order = Order(
                symbol=f"{symbol}_PUT",
                quantity=current_position.quantity,
                order_type=OrderType.BUY_TO_OPEN,
                price=data['put_option_strike'],
                timestamp=data['timestamp']
            )
            orders.append(put_option_order)
        
        return orders

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        signals = super().generate_signals(data)
        
        # Add logic for put option hedging
        current_price = data['close'].iloc[-1]
        put_strike = current_price * self.put_option_strike_pct
        put_signal = (signals[f'SMA_Crossover_sell'] == 1) | (data['close'] < data[f'SMA_{self.sma_window}'])
        
        signals = signals.with_columns([
            pl.Series(name='put_option_signal', values=put_signal),
            pl.Series(name='put_option_strike', values=[put_strike] * len(data)),
            pl.Series(name='put_option_expiry', values=[self.put_option_expiry_days] * len(data))
        ])
        
        return signals