# test_simulator.py

import asyncio
from simulator import Simulator
from data_streamer import StdoutStreamer
import polars as pl

async def main():
    # Initialize the simulator with a config file
    simulator = Simulator('quantaforge/simulator/config/simulation_config.yaml')

    # Replace the default Kafka streamer with StdoutStreamer for testing
    simulator.data_streamer = StdoutStreamer()

    # Test 1: Run a batch simulation for multiple symbols
    print("Test 1: Batch simulation for multiple symbols")
    results = simulator.run(
        symbols=['AAPL', 'GOOGL', 'MSFT'],
        method='garch',
        mode='batch',
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    for symbol, data in results:
        print(f"\nResults for {symbol}:")
        print(data.head())

    # Test 2: Run a deterministic simulation
    print("\nTest 2: Deterministic simulation")
    det_results = simulator.run(
        symbols=['TSLA'],
        method='deterministic',
        mode='batch',
        start_date='2023-01-01',
        end_date='2023-03-31',
        price_adjustment=1.1  # 10% price increase
    )
    
    for symbol, data in det_results:
        print(f"\nDeterministic results for {symbol}:")
        print(data.head())

    # Test 3: Run different statistical simulations
    print("\nTest 3: Different statistical simulations")
    stat_methods = ['garch', 'mean_reversion', 'jump_diffusion']
    for method in stat_methods:
        stat_results = simulator.run(
            symbols=['AMZN'],
            method=method,
            mode='batch',
            start_date='2023-01-01',
            end_date='2023-06-30'
        )
        
        for symbol, data in stat_results:
            print(f"\n{method.capitalize()} simulation results for {symbol}:")
            print(data.head())

    # Test 4: Realtime streaming simulation
    print("\nTest 4: Realtime streaming simulation")
    stream_results = simulator.run(
        symbols=['NVDA'],
        method='garch',
        mode='realtime',
        speed=10.0,  # 10x speed
        start_date='2023-01-01',
        end_date='2023-01-31'  # One month of data for quicker demonstration
    )
    
    print("Streaming realtime simulation results (press Ctrl+C to stop):")
    try:
        await simulator.stream_results(stream_results, mode='realtime', speed=10.0, time_dilation='daily')
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")

    # Test 5: Fast streaming of pre-computed results
    print("\nTest 5: Fast streaming of pre-computed results")
    fast_results = simulator.run(
        symbols=['META'],
        method='mean_reversion',
        mode='batch',
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    print("Fast streaming of pre-computed results:")
    await simulator.stream_results(fast_results, mode='fast')

if __name__ == "__main__":
    asyncio.run(main())