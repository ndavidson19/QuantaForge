from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pypfopt import (
    expected_returns,
    risk_models,
    EfficientFrontier,
    BlackLittermanModel,
    HRPOpt,
    CLA,
    objective_functions,
    plotting
)
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.efficient_semivariance import EfficientSemivariance
from pypfopt.efficient_cvar import EfficientCVaR
from quantforge import QuantForgeAPI
import redis
import json
import pickle
from datetime import timedelta

app = FastAPI(title="PyPortfolioOpt Complete API with Caching")

API_KEY = APIKeyHeader(name="X-API-Key")

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_api_key(api_key: str = Depends(API_KEY)):
    if api_key != "your_secret_api_key":
        raise HTTPException(status_code=403, detail="Could not validate API Key")
    return api_key

# ... (previous model definitions remain the same)

qf_api = QuantForgeAPI("your_api_key")

def cache_key(prefix: str, **kwargs) -> str:
    """Generate a cache key based on the function arguments."""
    key_parts = [prefix] + [f"{k}:{v}" for k, v in sorted(kwargs.items())]
    return ":".join(key_parts)

def cache_result(key: str, result: Any, expiration: int = 3600):
    """Cache the result using Redis."""
    redis_client.set(key, pickle.dumps(result), ex=expiration)

def get_cached_result(key: str) -> Optional[Any]:
    """Retrieve cached result from Redis."""
    cached = redis_client.get(key)
    if cached:
        return pickle.loads(cached)
    return None

async def get_historical_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical prices with caching."""
    cache_key = f"prices:{','.join(sorted(tickers))}:{start_date}:{end_date}"
    cached_prices = get_cached_result(cache_key)
    if cached_prices is not None:
        return cached_prices
    
    prices = await qf_api.get_historical_prices(tickers, start_date, end_date)
    cache_result(cache_key, prices, expiration=86400)  # Cache for 24 hours
    return prices

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest, api_key: str = Depends(get_api_key)):
    cache_key = f"optimize:{hash(frozenset(request.dict().items()))}"
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return OptimizationResponse(**cached_result)

    try:
        prices = await get_historical_prices(request.tickers, request.start_date, request.end_date)
        
        # Calculate expected returns
        mu = getattr(expected_returns, request.expected_returns_method)(prices)
        
        # Calculate risk model
        if request.risk_model_method == "ledoit_wolf":
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        else:
            S = getattr(risk_models, request.risk_model_method)(prices)
        
        # Initialize optimizer
        if request.optimization_method in ["max_sharpe", "min_volatility", "max_quadratic_utility", "efficient_risk", "efficient_return"]:
            ef = EfficientFrontier(mu, S, weight_bounds=request.weight_bounds)
        elif request.optimization_method == "efficient_semivariance":
            ef = EfficientSemivariance(mu, returns=prices.pct_change().dropna(), weight_bounds=request.weight_bounds)
        elif request.optimization_method == "efficient_cvar":
            ef = EfficientCVaR(mu, returns=prices.pct_change().dropna(), weight_bounds=request.weight_bounds)
        else:
            raise ValueError(f"Unknown optimization method: {request.optimization_method}")
        
        # Add constraints and objectives
        if request.constraints:
            for constraint in request.constraints:
                ef.add_constraint(constraint)
        
        if request.additional_objectives:
            for obj in request.additional_objectives:
                obj_func = getattr(objective_functions, obj["name"])
                ef.add_objective(obj_func, **{k: v for k, v in obj.items() if k != "name"})
        
        # Optimize
        weights = getattr(ef, request.optimization_method)(**request.additional_parameters or {})
        
        # Get performance
        performance = ef.portfolio_performance()
        
        result = OptimizationResponse(
            weights=weights,
            performance={
                "expected_return": performance[0],
                "volatility": performance[1],
                "sharpe_ratio": performance[2]
            }
        )
        
        cache_result(cache_key, result.dict(), expiration=3600)  # Cache for 1 hour
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/black-litterman", response_model=OptimizationResponse)
async def black_litterman_optimize(request: BlackLittermanRequest, api_key: str = Depends(get_api_key)):
    cache_key = f"black_litterman:{hash(frozenset(request.dict().items()))}"
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return OptimizationResponse(**cached_result)

    try:
        prices = await get_historical_prices(request.tickers, request.start_date, request.end_date)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        
        delta = BlackLittermanModel.market_implied_risk_aversion(request.market_prices)
        prior = BlackLittermanModel.market_implied_prior_returns(request.mcaps, delta, S)
        
        bl = BlackLittermanModel(S, pi=prior, absolute_views=request.absolute_views, tau=request.tau, omega=request.omega)
        
        if request.relative_views:
            for view in request.relative_views:
                bl.add_relative_view(view["assets"], view["weights"], view["confidence"])
        
        bl_returns = bl.bl_returns()
        ef = EfficientFrontier(bl_returns, S)
        weights = ef.max_sharpe()
        performance = ef.portfolio_performance()
        
        result = OptimizationResponse(
            weights=weights,
            performance={
                "expected_return": performance[0],
                "volatility": performance[1],
                "sharpe_ratio": performance[2]
            }
        )
        
        cache_result(cache_key, result.dict(), expiration=3600)  # Cache for 1 hour
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ... (implement caching for other endpoints similarly)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)