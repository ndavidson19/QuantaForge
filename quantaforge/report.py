import logging

class Report:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def generate(self):
        logging.debug("Generating report...")
        initial_cash = self.portfolio.initial_cash
        final_cash = self.portfolio.cash
        total_value = self.portfolio.value(current_prices=self.portfolio.data['close'].to_numpy())
        profit_loss = total_value - initial_cash
        num_trades = len(self.portfolio.history)
        
        print("===== Backtest Report =====")
        print(f"Initial Cash: ${initial_cash}")
        print(f"Final Cash: ${final_cash}")
        print(f"Total Portfolio Value: ${total_value}")
        print(f"Total Profit/Loss: ${profit_loss}")
        print(f"Number of Trades: {num_trades}")
        print("\nTrades:")
        for trade in self.portfolio.history:
            symbol, quantity, price, action = trade
            print(f"{action.capitalize()} {quantity} {symbol} at ${price}")
        print("===========================")
