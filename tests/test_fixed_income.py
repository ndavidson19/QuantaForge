import unittest
import pandas as pd
import numpy as np
import logging
from examples.FixedIncomeYieldStrategy import FixedIncomeYieldStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFixedIncomeYieldStrategy(unittest.TestCase):
    def setUp(self):
        self.bond_universe = ['T10Y', 'T30Y', 'AAPL30', 'MSFT25', 'JNJ20', 'HYG', 'LQD']
        self.strategy = FixedIncomeYieldStrategy(
            self.bond_universe,
            target_duration=7,
            max_position_size=0.2,
            min_credit_rating='BBB-',
            rebalance_frequency='monthly'
        )
        self.mock_data = self._create_mock_data()

    def _create_mock_data(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        mock_data = pd.DataFrame(index=dates)
        
        mock_data['Treasury_Curve'] = [
            {'3M': 0.01, '2Y': 0.02, '5Y': 0.03, '10Y': 0.04, '30Y': 0.05}
            for _ in range(len(dates))
        ]
        
        for bond in self.bond_universe:
            mock_data[f"{bond}_Yield"] = np.random.uniform(0.01, 0.06, len(dates))
            mock_data[f"{bond}_Duration"] = np.random.uniform(1, 20, len(dates))
        
        return mock_data

    def test_analyze_yield_curve(self):
        logger.info('Testing analyze_yield_curve method.')
        curve_analysis = self.strategy.analyze_yield_curve(self.mock_data)
        logger.info(f'Yield curve analysis results: {curve_analysis}')
        self.assertIn('slope_2_10', curve_analysis)
        self.assertIn('slope_3m_10y', curve_analysis)
        self.assertIn('is_inverted', curve_analysis)

    def test_calculate_yield_spreads(self):
        logger.info('Testing calculate_yield_spreads method.')
        spreads = self.strategy.calculate_yield_spreads(self.mock_data)
        logger.info(f'Yield spreads: {spreads}')
        self.assertGreater(len(spreads), 0)
        self.assertNotIn('T10Y', spreads)  # Treasury bonds should not be in spreads
        self.assertIn('AAPL30', spreads)

    def test_rank_bonds(self):
        logger.info('Testing rank_bonds method.')
        ranked_bonds = self.strategy.rank_bonds(self.mock_data)
        logger.info(f'Ranked bonds: {ranked_bonds}')
        self.assertGreater(len(ranked_bonds), 0)
        self.assertLessEqual(len(ranked_bonds), len(self.bond_universe))
        if len(ranked_bonds) > 1:
            self.assertGreaterEqual(ranked_bonds[0][1], ranked_bonds[-1][1])  # Check if sorted by yield

    def test_generate_signals(self):
        logging.info("Testing generate_signals method.")
        signals = self.strategy.generate_signals(self.mock_data)
        logging.info(f"Generated signals: {signals}")
        self.assertIsInstance(signals, dict)
        self.assertLessEqual(len(signals), len(self.bond_universe))
        self.assertGreater(len(signals), 0, "No signals were generated")
        for bond, signal in signals.items():
            self.assertIn(f"{bond}_buy", signal.columns)
            self.assertIn(f"{bond}_ytm", signal.columns)
            self.assertIn(f"{bond}_duration", signal.columns)
            self.assertTrue((signal[f"{bond}_buy"] > 0).any(), f"No buy signal for {bond}")

    def test_manage_risk(self):
        logging.info("Testing manage_risk method.")
        risk_signals = self.strategy.manage_risk(self.mock_data)
        logging.info(f"Risk signals: {risk_signals}")
        self.assertIsInstance(risk_signals, dict)
        if risk_signals:  # Only check if risk signals were generated
            for bond, signal in risk_signals.items():
                self.assertTrue(f"{bond}_buy" in signal.columns or f"{bond}_sell" in signal.columns)
                self.assertIn(f"{bond}_duration", signal.columns)
                if f"{bond}_buy" in signal.columns:
                    self.assertTrue((signal[f"{bond}_buy"] > 0).any(), f"No buy risk signal for {bond}")
                if f"{bond}_sell" in signal.columns:
                    self.assertTrue((signal[f"{bond}_sell"] > 0).any(), f"No sell risk signal for {bond}")

if __name__ == '__main__':
    unittest.main()