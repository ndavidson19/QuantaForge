from sqlite3 import DataError
from confluent_kafka import Consumer, KafkaException
import json

class DataFeed:
    def __init__(self, topic, bootstrap_servers, group_id='quanta_group'):
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        })
        self.consumer.subscribe([topic])

    def get_data(self):
        while True:
            msg = self.consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == DataError._PARTITION_EOF:
                    continue
                else:
                    raise KafkaException(msg.error())
            yield json.loads(msg.value().decode('utf-8'))

    def close(self):
        self.consumer.close()
