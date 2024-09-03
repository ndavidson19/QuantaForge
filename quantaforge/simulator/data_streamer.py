import json
import time
import asyncio
import sys
import polars as pl
from abc import ABC, abstractmethod
from typing import Any
from kafka import KafkaProducer
from datetime import date, datetime

class DataStreamerBase(ABC):
    @abstractmethod
    async def stream_data_fast(self, topic: str, data: pl.DataFrame) -> None:
        pass

    @abstractmethod
    async def stream_data_realtime(self, topic: str, data: pl.DataFrame, speed: float, time_dilation: str) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

class DataStreamer(DataStreamerBase):
    def __init__(self, bootstrap_servers=['localhost:9092']):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=self._json_serializer).encode('utf-8')
        )

    async def stream_data_fast(self, topic: str, data: pl.DataFrame) -> None:
        for row in data.iter_rows(named=True):
            message = dict(row)
            self.producer.send(topic, value=message)
        self.producer.flush()

    async def stream_data_realtime(self, topic: str, data: pl.DataFrame, speed: float, time_dilation: str) -> None:
        time_column = 'date'
        data = data.sort(time_column)
        
        for i in range(1, len(data)):
            current_row = data.row(i, named=True)
            previous_row = data.row(i-1, named=True)
            
            time_diff = self._calculate_time_diff(current_row[time_column], previous_row[time_column], time_dilation)
            sleep_time = time_diff / speed
            
            await asyncio.sleep(sleep_time)
            
            message = dict(current_row)
            self.producer.send(topic, value=message)
        
        self.producer.flush()

    def _calculate_time_diff(self, current_time, previous_time, time_dilation):
        if time_dilation == 'second':
            return (current_time - previous_time).total_seconds()
        elif time_dilation == 'daily':
            return (current_time - previous_time).days
        elif time_dilation == 'monthly':
            return (current_time.year - previous_time.year) * 12 + current_time.month - previous_time.month
        else:
            raise ValueError(f"Unsupported time dilation: {time_dilation}")
    def _json_serializer(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    async def close(self) -> None:
        self.producer.close()

class StdoutStreamer(DataStreamerBase):
    async def stream_data_fast(self, topic: str, data: pl.DataFrame) -> None:
        for row in data.iter_rows(named=True):
            message = dict(row)
            print(json.dumps({"topic": topic, "data": message}, default=self._json_serializer), flush=True)

    async def stream_data_realtime(self, topic: str, data: pl.DataFrame, speed: float, time_dilation: str) -> None:
        time_column = 'date'
        data = data.sort(time_column)
        
        for i in range(1, len(data)):
            current_row = data.row(i, named=True)
            previous_row = data.row(i-1, named=True)
            
            time_diff = self._calculate_time_diff(current_row[time_column], previous_row[time_column], time_dilation)
            sleep_time = time_diff / speed
            
            await asyncio.sleep(sleep_time)
            
            message = dict(current_row)
            print(json.dumps({"topic": topic, "data": message}, default=self._json_serializer), flush=True)

    def _calculate_time_diff(self, current_time, previous_time, time_dilation):
        if time_dilation == 'second':
            return (current_time - previous_time).total_seconds()
        elif time_dilation == 'daily':
            return (current_time - previous_time).days
        elif time_dilation == 'monthly':
            return (current_time.year - previous_time.year) * 12 + current_time.month - previous_time.month
        else:
            raise ValueError(f"Unsupported time dilation: {time_dilation}")
        
    def _json_serializer(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    async def close(self) -> None:
        sys.stdout.flush()