class Indicator:
    def __init__(self, name, window):
        self.name = name
        self.window = window

    def calculate(self, data):
        if self.name == 'SMA':
            return data['close'].rolling(window=self.window).mean()
        # Add more indicators as needed
