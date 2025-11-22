import redis
import json
from typing import Optional, Any
from config import settings

class RedisCache:
    def __init__(self):
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )
    
    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value di cache dengan expiration"""
        try:
            serialized = json.dumps(value)
            return self.client.setex(key, expire, serialized)
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value dari cache"""
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete key dari cache"""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys yang match pattern"""
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Redis clear_pattern error: {e}")
            return 0
    
    def ping(self) -> bool:
        """Check Redis connection"""
        try:
            return self.client.ping()
        except Exception:
            return False