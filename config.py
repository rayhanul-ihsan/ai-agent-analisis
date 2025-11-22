from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # LLM Configuration
    llm_api_key: str
    llm_base_url: str
    
    # Embedding Configuration
    embedding_base_url: str
    
    # Vector DB Configuration
    vector_db: Literal["chroma", "faiss"] = "chroma"
    
    # Database Configuration (SQLite)
    database_path: str = "./data/documents.db"
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Application Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()