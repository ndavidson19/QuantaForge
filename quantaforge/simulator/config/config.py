from pydantic import BaseSettings

class Settings(BaseSettings):
    kafka_bootstrap_servers: str = "localhost:9092"
    data_dir: str = "data/historical"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"