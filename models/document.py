from sqlalchemy import Column, String, DateTime, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    content = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic Models untuk API
class DocumentMetadata(BaseModel):
    id: str
    filename: str
    file_type: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_extracted: int
    message: str

class IndexResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    message: str

class ChatRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    doc_ids: List[str]