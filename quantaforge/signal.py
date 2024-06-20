import polars as pl

class Signal:
    def __init__(self, name, indicator, reference):
        self.name = name
        self.indicator = indicator
        self.reference = reference

    def get_indicator_column_name(self):
        if self.indicator == 'SMA':
            return f'SMA_{self.reference}'
        return self.indicator

    def generate(self, data, indicator_data):
        if self.name == 'CrossOver':
            indicator_col = f'SMA_{self.reference}'  # Ensure the column name is correct
            ref_col = self.reference

            # Extract the columns as Series and ensure they are the same shape
            indicator_series = indicator_data[indicator_col].to_numpy().flatten()
            ref_series = data['close'].to_numpy().flatten()

            # Perform the comparison
            signal_series = (indicator_series > ref_series).astype(int)

            # Create a DataFrame with the signal
            signal_df = pl.DataFrame({'signal': signal_series})

            return signal_df

        # Add more signals as needed
        raise ValueError(f"Unsupported signal: {self.name}")
    
