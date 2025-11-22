import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict

class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.Client(ChromaSettings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_embeddings(
        self, 
        doc_id: str, 
        embeddings: List[List[float]], 
        chunks: List[str]
    ) -> int:
        """Menambahkan embeddings ke vector store"""
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        metadatas = [
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "text": chunk[:500]
            }
            for i, chunk in enumerate(chunks)
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 3,
        doc_ids: List[str] = None
    ) -> List[Dict]:
        """Mencari chunks yang paling similar dengan query"""
        where_filter = None
        if doc_ids:
            where_filter = {"doc_id": {"$in": doc_ids}}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        
        formatted_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        
        return formatted_results
    
    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks dari dokumen tertentu"""
        results = self.collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
    
    def get_stats(self) -> Dict:
        """Get statistics tentang vector store"""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name
        }