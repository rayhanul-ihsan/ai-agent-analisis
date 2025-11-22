import httpx
from typing import List
from config import settings

class Embedder:
    def __init__(self):
        self.base_url = settings.embedding_base_url
        self.model = "ebbge-m3"
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed single text menjadi vector"""
        result = await self.embed_batch([text])
        return result[0]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts sekaligus"""
        embeddings = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for text in texts:
                try:
                    response = await client.post(
                        self.base_url,
                        json={
                            "model": self.model,
                            "input": text
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if "data" in data and len(data["data"]) > 0:
                        embedding = data["data"][0]["embedding"]
                        embeddings.append(embedding)
                    else:
                        raise ValueError("Invalid response format from embedding API")
                        
                except httpx.HTTPError as e:
                    raise Exception(f"Error calling embedding API: {str(e)}")
                except Exception as e:
                    raise Exception(f"Error processing embedding: {str(e)}")
        
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed query untuk search"""
        return await self.embed_text(query)
    