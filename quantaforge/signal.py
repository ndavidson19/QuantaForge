class Signal:
    def __init__(self, name, indicator, reference):
        self.name = name
        self.indicator = indicator
        self.reference = reference

    def generate(self, data, indicator_data):
        if self.name == 'CrossOver':
            # Simple crossover signal
            return indicator_data > data[self.reference]
        # Add more signals as needed
