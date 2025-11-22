from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uuid
from typing import Optional, List

from config import settings
from models.document import (
    Document, DocumentMetadata, UploadResponse, 
    IndexResponse, ChatRequest, ChatResponse
)
from services.extractor import DocumentExtractor
from services.embedder import Embedder
from services.llm_client import LLMClient
from services.retriever import Retriever
from db.sqlite_conn import get_db_session, init_db, get_db
from db.vector_store import VectorStore
from db.redis_cache import RedisCache

# Initialize FastAPI
app = FastAPI(
    title="Document Q&A API",
    description="RAG-based Document Question Answering System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
extractor = DocumentExtractor(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)
embedder = Embedder()
llm_client = LLMClient()
retriever = Retriever()
vector_store = VectorStore()
redis_cache = RedisCache()

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    print("✅ Database initialized")
    print(f"✅ Vector DB: {settings.vector_db}")
    print(f"✅ Chunk size: {settings.chunk_size}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Document Q&A API is running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    redis_status = redis_cache.ping()
    vector_stats = vector_store.get_stats()
    
    return {
        "status": "healthy",
        "redis": "connected" if redis_status else "disconnected",
        "vector_store": vector_stats,
        "settings": {
            "chunk_size": settings.chunk_size,
            "top_k": settings.top_k_results
        }
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session)
):
    """
    Upload dan ekstraksi dokumen
    
    Supported formats: PDF, DOCX, CSV
    """
    try:
        # Validasi file type
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["pdf", "docx", "csv"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Extract text
        text = extractor.extract(file_content, file_extension)
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document"
            )
        
        # Chunk text
        chunks = extractor.chunk_text(text)
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Save to database
        doc = Document(
            id=doc_id,
            filename=file.filename,
            file_type=file_extension,
            content=text[:10000],  # Save preview
            chunk_count=len(chunks)
        )
        db.add(doc)
        db.commit()
        
        # Cache chunks untuk indexing
        redis_cache.set(
            f"chunks:{doc_id}", 
            chunks, 
            expire=3600  # 1 hour
        )
        
        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunks_extracted=len(chunks),
            message=f"Document uploaded successfully. Use /index endpoint to index the document."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")
    finally:
        db.close()

@app.post("/index", response_model=IndexResponse)
async def index_document(doc_id: str, db: Session = Depends(get_db_session)):
    """
    Index dokumen ke vector store
    
    Args:
        doc_id: Document ID dari /upload
    """
    try:
        # Get document dari database
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks dari cache
        chunks = redis_cache.get(f"chunks:{doc_id}")
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Chunks not found. Please upload the document again."
            )
        
        # Generate embeddings
        embeddings = await embedder.embed_batch(chunks)
        
        # Add to vector store
        chunks_indexed = vector_store.add_embeddings(doc_id, embeddings, chunks)
        
        # Clear cache
        redis_cache.delete(f"chunks:{doc_id}")
        
        return IndexResponse(
            doc_id=doc_id,
            chunks_indexed=chunks_indexed,
            message=f"Document indexed successfully with {chunks_indexed} chunks"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")
    finally:
        db.close()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Q&A berbasis konteks dokumen
    
    Args:
        question: User question
        doc_ids: Optional list of document IDs to search in
    """
    try:
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Retrieve relevant chunks
        context_chunks = await retriever.retrieve(
            query=request.question,
            doc_ids=request.doc_ids,
            top_k=settings.top_k_results
        )
        
        if not context_chunks:
            return ChatResponse(
                answer="I couldn't find any relevant information in the documents to answer your question.",
                sources=[],
                doc_ids=[]
            )
        
        # Get answer from LLM
        answer = await llm_client.ask_llm(request.question, context_chunks)
        
        # Format sources
        sources = [
            {
                "chunk_id": chunk["id"],
                "doc_id": chunk["metadata"]["doc_id"],
                "text_preview": chunk["text"][:200] + "...",
                "relevance_score": 1 - chunk["distance"] if chunk["distance"] else None
            }
            for chunk in context_chunks
        ]
        
        # Get unique doc_ids
        doc_ids_used = list(set([chunk["metadata"]["doc_id"] for chunk in context_chunks]))
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            doc_ids=doc_ids_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/docs/{doc_id}", response_model=DocumentMetadata)
async def get_document(doc_id: str, db: Session = Depends(get_db_session)):
    """
    Get metadata dokumen
    
    Args:
        doc_id: Document ID
    """
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentMetadata(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
            updated_at=doc.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        db.close()

@app.get("/docs")
async def list_documents(
    skip: int = 0, 
    limit: int = 10,
    db: Session = Depends(get_db_session)
):
    """List all documents"""
    try:
        docs = db.query(Document).offset(skip).limit(limit).all()
        total = db.query(Document).count()
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "documents": [
                DocumentMetadata(
                    id=doc.id,
                    filename=doc.filename,
                    file_type=doc.file_type,
                    chunk_count=doc.chunk_count,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at
                )
                for doc in docs
            ]
        }
    finally:
        db.close()

@app.delete("/docs/{doc_id}")
async def delete_document(doc_id: str, db: Session = Depends(get_db_session)):
    """Delete document dan embeddings-nya"""
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete dari vector store
        vector_store.delete_by_doc_id(doc_id)
        
        # Delete dari database
        db.delete(doc)
        db.commit()
        
        # Clear related cache
        redis_cache.clear_pattern(f"*{doc_id}*")
        
        return {"message": "Document deleted successfully", "doc_id": doc_id}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)