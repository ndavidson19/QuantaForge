from pypfopt.black_litterman import BlackLittermanModel
import pandas as pd

class EnhancedBlackLittermanModel(BlackLittermanModel):
    def __init__(self, cov_matrix, pi="equal", absolute_views=None, **kwargs):
        super().__init__(cov_matrix, pi=pi, absolute_views=absolute_views, **kwargs)

    def optimize(self, risk_aversion=1, method="max_sharpe", **kwargs):
        bl_returns = self.bl_returns()
        bl_cov = self.bl_cov()
        ef = EnhancedEfficientFrontier(bl_returns, bl_cov)
        return ef.optimize(method=method, **kwargs)