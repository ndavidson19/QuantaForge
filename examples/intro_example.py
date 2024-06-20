import quantaforge as qf

# Create a new strategy
strategy = qf.Strategy()

# Add a new indicator
strategy.add_indicator(qf.Indicator('SMA', 5))

# Add a new signal
strategy.add_signal(qf.Signal('CrossOver', 'SMA', 5))

# Add a new position
strategy.add_position(qf.Position('Long', 1))

# Add a new risk management
strategy.add_risk_management(qf.RiskManagement('StopLoss', 0.02))

# Add a new portfolio
portfolio = qf.Portfolio('SimplePortfolio', 100000)
strategy.add_portfolio(portfolio)

# Add a new backtest
backtest = qf.Backtest(name='SimpleBacktest', strategy=strategy, portfolio=portfolio)

# Mock data to emulate a 5 day SMA crossover strategy

data = {
    'symbol': ['AAPL'] * 20,
    'close': [150, 152, 148, 151, 155, 160, 158, 162, 165, 170, 175, 180, 190, 210, 200, 190, 180, 170, 160, 150]
}

# Set the data for the backtest
backtest.set_data(data)
strategy.add_backtest(backtest)

# Add a new report
report = qf.Report(portfolio=portfolio)
strategy.add_report(report)

# Run the strategy
strategy.run()
strategy.report.generate()
