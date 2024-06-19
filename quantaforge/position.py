class Position:
    def __init__(self, type, size):
        self.type = type
        self.size = size

    def execute(self, signal):
        if signal and self.type == 'Long':
            return 'Buy', self.size
        elif not signal and self.type == 'Short':
            return 'Sell', self.size
        return None, 0
