import httpx
from typing import List, Dict
from config import settings

class LLMClient:
    def __init__(self):
        self.base_url = settings.llm_base_url
        self.api_key = settings.llm_api_key
        self.model = "gpt-oss-20b"
    
    async def ask_llm(self, question: str, context_chunks: List[Dict]) -> str:
        """Mengirim pertanyaan ke LLM dengan konteks dari dokumen"""
        context = "\n\n".join([
            f"[Source {i+1}]:\n{chunk['text']}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""You are a document assistant. Use the context below to answer the user's question.
If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."
Always cite the source number when referencing information.

Context:
{context}

Question:
{question}

Answer:"""
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system", 
                                "content": "You are a helpful assistant that answers questions based on document context."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1000
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    answer = data["choices"][0]["message"]["content"]
                    return answer.strip()
                else:
                    raise ValueError("Invalid response format from LLM API")
                    
            except httpx.HTTPError as e:
                raise Exception(f"Error calling LLM API: {str(e)}")
            except Exception as e:
                raise Exception(f"Error processing LLM response: {str(e)}")