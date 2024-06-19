class RiskManagement:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def apply(self, data):
        if self.type == 'StopLoss':
            return data['close'] * (1 - self.value)
        # Add more risk management strategies as needed
