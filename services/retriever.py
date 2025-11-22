from typing import List, Dict, Optional
from services.embedder import Embedder
from db.vector_store import VectorStore
from db.redis_cache import RedisCache
from config import settings
import hashlib

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.cache = RedisCache()
        self.top_k = settings.top_k_results
    
    def _generate_cache_key(self, query: str, doc_ids: Optional[List[str]] = None) -> str:
        """Generate cache key untuk query"""
        cache_str = f"{query}_{doc_ids}" if doc_ids else query
        return f"query:{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    async def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        doc_ids: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """Retrieve relevant chunks untuk query"""
        k = top_k or self.top_k
        
        if use_cache:
            cache_key = self._generate_cache_key(query, doc_ids)
            cached_results = self.cache.get(cache_key)
            if cached_results:
                print(f"Cache hit for query: {query[:50]}")
                return cached_results
        
        query_embedding = await self.embedder.embed_query(query)
        
        results = self.vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=k,
            doc_ids=doc_ids
        )
        
        if use_cache and results:
            cache_key = self._generate_cache_key(query, doc_ids)
            self.cache.set(cache_key, results, expire=1800)
        
        return results