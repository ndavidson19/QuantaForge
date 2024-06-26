import polars as pl
import numpy as np
from typing import Dict, Any, List, Union, Callable
from datetime import datetime
from quantaforge.generics import Portfolio, StrategyBase
from quantaforge.order import Order, OrderType
from quantaforge.performance_metrics import PerformanceMetrics


class PositionSizer:
    @staticmethod
    def kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
        return win_rate - ((1 - win_rate) / win_loss_ratio)

    @staticmethod
    def volatility_sizing(volatility: float, risk_per_trade: float, account_size: float) -> float:
        return (risk_per_trade * account_size) / volatility

class MonteCarloSimulation:
    def __init__(self, returns: np.array, num_simulations: int = 1000, time_horizon: int = 252):
        self.returns = returns
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon

    def run(self) -> np.array:
        simulations = np.zeros((self.time_horizon, self.num_simulations))
        for i in range(self.num_simulations):
            simulations[:, i] = np.cumprod(np.random.choice(self.returns, self.time_horizon) + 1)
        return simulations

class AdvancedRiskManagement:
    @staticmethod
    def calculate_var(returns: np.array, confidence_level: float = 0.95) -> float:
        return np.percentile(returns, 100 * (1 - confidence_level))

    @staticmethod
    def calculate_cvar(returns: np.array, confidence_level: float = 0.95) -> float:
        var = AdvancedRiskManagement.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

class ParameterOptimizer:
    def __init__(self, backtest_func: Callable, param_grid: Dict[str, List[Any]], optimization_metric: str):
        self.backtest_func = backtest_func
        self.param_grid = param_grid
        self.optimization_metric = optimization_metric

    def optimize(self) -> Dict[str, Any]:
        best_params = None
        best_metric = float('-inf') if self.optimization_metric != 'max_drawdown' else float('inf')

        for params in self._generate_param_combinations():
            results = self.backtest_func(**params)
            current_metric = results[self.optimization_metric]

            if (self.optimization_metric != 'max_drawdown' and current_metric > best_metric) or \
               (self.optimization_metric == 'max_drawdown' and current_metric < best_metric):
                best_metric = current_metric
                best_params = params

        return {'best_params': best_params, 'best_metric': best_metric}

    def _generate_param_combinations(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))


class Backtest:
    def __init__(self, 
                 name: str, 
                 strategy: StrategyBase, 
                 initial_capital: float,
                 commission: Union[float, Callable] = 0.001,
                 slippage: Union[float, Callable] = 0.0005,
                 margin_requirement: float = 1.0,
                 interest_rate: float = 0.0,
                 data_frequency: str = '1d',
                 position_sizer: Callable = None,
                 risk_manager: Callable = None
                 ):
        self.name = name
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.margin_requirement = margin_requirement
        self.interest_rate = interest_rate
        self.data_frequency = data_frequency
        self.portfolio = Portfolio(initial_capital)
        self.data: pl.DataFrame = None
        self.results: Dict[str, Any] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.position_sizer = position_sizer or PositionSizer.volatility_sizing
        self.risk_manager = risk_manager or AdvancedRiskManagement.calculate_var
        
    def set_data(self, data: pl.DataFrame):
        self.data = data
        self._validate_data()

    def _validate_data(self):
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {', '.join(missing_columns)}")

    def run(self):
        if self.data is None:
            raise ValueError("Data must be set before running backtest")

        # Calculate indicators
        for indicator in self.strategy.indicators.values():
            self.data = indicator.calculate(self.data)
        
        # Position sizing
        position_size = self.position_sizer(volatility, risk_per_trade, self.initial_capital)

        # Risk management
        var = self.risk_manager(returns)

        # Generate signals
        signals = self.strategy.generate_signals(self.data)

        # Combine data and signals
        self.data = self.data.hstack(signals)

        # Sort data by timestamp
        self.data = self.data.sort('timestamp')

        # Run the backtest
        for row in self.data.iter_rows(named=True):
            self._process_bar(row)

        self.calculate_performance_metrics()

    def _process_bar(self, row: Dict[str, Any]):
        timestamp = row['timestamp']
        symbol = row['symbol']

        # Update portfolio value
        self._update_portfolio_value(row)

        # Process pending orders
        self._process_pending_orders(row)

        # Check for exit signals
        for position in self.portfolio.positions.values():
            if self._check_exit_conditions(row, position.symbol):
                self._place_order(symbol, -position.quantity, OrderType.MARKET, row)

        # Check for entry signals
        if self._check_entry_conditions(row):
            quantity = self._calculate_position_size(row)
            self._place_order(symbol, quantity, OrderType.MARKET, row)

        # Apply risk management
        for risk_rule in self.strategy.risk_management:
            risk_rule.apply(self.portfolio, row)

        # Record equity
        self.equity_curve.append(self.portfolio.total_value)

    def _check_entry_conditions(self, row: Dict[str, Any]) -> bool:
        return all(condition.evaluate(row) for condition in self.strategy.entry_conditions)

    def _check_exit_conditions(self, row: Dict[str, Any], symbol: str) -> bool:
        return all(condition.evaluate(row) for condition in self.strategy.exit_conditions)

    def _calculate_position_size(self, row: Dict[str, Any]) -> int:
        # Implement sophisticated position sizing here
        # This is a simple example using a fixed percentage of portfolio value
        portfolio_value = self.portfolio.total_value
        price = row['close']
        return int((portfolio_value * 0.01) / price)  # 1% of portfolio value

    def _place_order(self, symbol: str, quantity: int, order_type: OrderType, row: Dict[str, Any]):
        price = self._calculate_execution_price(row, quantity, order_type)
        order = Order(symbol, quantity, order_type, price, row['timestamp'])
        self.orders.append(order)
        self._execute_order(order, row)

    def _calculate_execution_price(self, row: Dict[str, Any], quantity: int, order_type: OrderType) -> float:
        base_price = row['close']
        
        # Apply slippage
        slippage = self.slippage if isinstance(self.slippage, float) else self.slippage(quantity)
        slippage_adjustment = base_price * slippage * (1 if quantity > 0 else -1)
        
        return base_price + slippage_adjustment

    def _execute_order(self, order: Order, row: Dict[str, Any]):
        commission = self._calculate_commission(order)
        total_cost = order.quantity * order.price + commission

        if order.quantity > 0:  # Buy order
            if self.portfolio.cash >= total_cost:
                self.portfolio.open_position(order.symbol, order.quantity, order.price, order.timestamp)
                self.portfolio.cash -= total_cost
            else:
                # Handle insufficient funds (e.g., partial fill or reject)
                pass
        else:  # Sell order
            self.portfolio.close_position(order.symbol, abs(order.quantity), order.price, order.timestamp)
            self.portfolio.cash += -total_cost  # Negative cost means cash inflow

        self._record_trade(order, commission)

    def _calculate_commission(self, order: Order) -> float:
        if callable(self.commission):
            return self.commission(order.quantity, order.price)
        return abs(order.quantity * order.price * self.commission)

    def _record_trade(self, order: Order, commission: float):
        self.trades.append({
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'quantity': order.quantity,
            'price': order.price,
            'commission': commission
        })

    def _update_portfolio_value(self, row: Dict[str, Any]):
        prices = {row['symbol']: row['close']}
        self.portfolio.update_value(row['timestamp'], prices)

    def _process_pending_orders(self, row: Dict[str, Any]):
        # Implement logic to process limit and stop orders
        pass

    def calculate_performance_metrics(self):
        metrics = PerformanceMetrics(self.equity_curve, self.trades, self.data_frequency)
        self.results = {
            'total_return': metrics.total_return(),
            'annualized_return': metrics.annualized_return(),
            'sharpe_ratio': metrics.sharpe_ratio(),
            'sortino_ratio': metrics.sortino_ratio(),
            'max_drawdown': metrics.max_drawdown(),
            'win_rate': metrics.win_rate(),
            'profit_factor': metrics.profit_factor(),
            'calmar_ratio': metrics.calmar_ratio(),
            'omega_ratio': metrics.omega_ratio(),
            'expectancy': metrics.expectancy(),
            'average_trade': metrics.average_trade(),
            'average_win': metrics.average_win(),
            'average_loss': metrics.average_loss(),
            'profit_factor': metrics.profit_factor(),
            'trade_count': len(self.trades),
        }

    def get_results(self) -> Dict[str, Any]:
        return self.results

    def plot_results(self):
        # Implement result plotting (you might want to use a library like matplotlib or plotly)
        pass

    def generate_report(self):
        # Implement a detailed report generation (e.g., PDF report with charts and statistics)
        pass

    def optimize_parameters(self, param_grid: Dict[str, List[Any]], optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        optimizer = ParameterOptimizer(self.run, param_grid, optimization_metric)
        return optimizer.optimize()
    
class PerformanceAnalyzer:
    def __init__(self, equity_curve: List[float], trades: List[Dict[str, Any]]):
        self.equity_curve = np.array(equity_curve)
        self.trades = trades
        self.returns = np.diff(self.equity_curve) / self.equity_curve[:-1]

    def calculate_metrics(self) -> Dict[str, float]:
        return {
            'total_return': self.calculate_total_return(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'var': AdvancedRiskManagement.calculate_var(self.returns),
            'cvar': AdvancedRiskManagement.calculate_cvar(self.returns),
        }

    def calculate_total_return(self) -> float:
        return (self.equity_curve[-1] / self.equity_curve[0]) - 1

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        excess_returns = self.returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        excess_returns = self.returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    def calculate_max_drawdown(self) -> float:
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak
        return drawdown.min()

    def calculate_win_rate(self) -> float:
        wins = sum(1 for trade in self.trades if trade['pnl'] > 0)
        return wins / len(self.trades)

    def calculate_profit_factor(self) -> float:
        gross_profits = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        gross_losses = sum(abs(trade['pnl']) for trade in self.trades if trade['pnl'] < 0)
        return gross_profits / gross_losses if gross_losses != 0 else float('inf')