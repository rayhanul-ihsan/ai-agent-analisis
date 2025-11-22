from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config import settings
from models.document import Base
from contextlib import contextmanager
import os

# Ensure data directory exists
os.makedirs(os.path.dirname(settings.database_path), exist_ok=True)

# Create SQLite database URL
DATABASE_URL = f"sqlite:///{settings.database_path}"

# Create engine with SQLite-specific settings
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},  # Allow multiple threads
    echo=False
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print(f"✅ SQLite database initialized at: {settings.database_path}")

@contextmanager
def get_db():
    """Get database session dengan context manager"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def get_db_session() -> Session:
    """Get database session untuk dependency injection"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass