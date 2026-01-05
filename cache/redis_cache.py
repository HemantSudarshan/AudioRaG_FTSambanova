"""
Redis Cache Implementation

High-performance caching with Redis.
"""

import json
import logging
import hashlib
from datetime import timedelta
from typing import Optional, Any, List
import pickle

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based cache for embeddings and query results.
    
    Provides reliable caching with TTL support and automatic serialization.
    """
    
    def __init__(
        self,
        url: str = "redis://localhost:6379",
        default_ttl: int = 3600,
        prefix: str = "audiorag:",
    ):
        """
        Initialize Redis cache.
        
        Args:
            url: Redis connection URL
            default_ttl: Default TTL in seconds
            prefix: Key prefix for namespacing
        """
        self.url = url
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._client = None
        self._connected = False
        
    @property
    def client(self):
        """Lazy connection to Redis."""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(
                    self.url,
                    decode_responses=False,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                self._client.ping()
                self._connected = True
                logger.info(f"Connected to Redis at {self.url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._connected = False
        return self._client
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._connected:
            return False
        try:
            self.client.ping()
            return True
        except Exception:
            self._connected = False
            return False
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self.prefix}{key}"
    
    def _hash_key(self, *args) -> str:
        """Create hash key from arguments."""
        data = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.is_connected:
            return None
        
        try:
            full_key = self._make_key(key)
            data = self.client.get(full_key)
            
            if data is None:
                logger.debug(f"Cache miss: {key}")
                return None
            
            logger.debug(f"Cache hit: {key}")
            return pickle.loads(data)
            
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        if not self.is_connected:
            return False
        
        try:
            full_key = self._make_key(key)
            data = pickle.dumps(value)
            ttl = ttl or self.default_ttl
            
            self.client.setex(full_key, ttl, data)
            logger.debug(f"Cache set: {key} (ttl={ttl}s)")
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        if not self.is_connected:
            return False
        
        try:
            full_key = self._make_key(key)
            result = self.client.delete(full_key)
            logger.debug(f"Cache delete: {key} (deleted={result})")
            return result > 0
            
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.is_connected:
            return False
        
        try:
            full_key = self._make_key(key)
            return self.client.exists(full_key) > 0
        except Exception:
            return False
    
    def get_embedding(self, text_hash: str) -> Optional[List[float]]:
        """
        Get cached embedding.
        
        Args:
            text_hash: Hash of the text
            
        Returns:
            Embedding vector or None
        """
        return self.get(f"embed:{text_hash}")
    
    def set_embedding(
        self,
        text_hash: str,
        embedding: List[float],
        ttl: int = 86400,  # 24 hours
    ) -> bool:
        """
        Cache an embedding.
        
        Args:
            text_hash: Hash of the text
            embedding: Embedding vector
            ttl: Time to live
            
        Returns:
            True if cached
        """
        return self.set(f"embed:{text_hash}", embedding, ttl)
    
    def get_query_result(
        self,
        query_hash: str,
        audio_id: str,
    ) -> Optional[str]:
        """
        Get cached query result.
        
        Args:
            query_hash: Hash of the query
            audio_id: ID of the audio
            
        Returns:
            Cached response or None
        """
        key = f"query:{audio_id}:{query_hash}"
        return self.get(key)
    
    def set_query_result(
        self,
        query_hash: str,
        audio_id: str,
        result: str,
        ttl: int = 3600,  # 1 hour
    ) -> bool:
        """
        Cache a query result.
        
        Args:
            query_hash: Hash of the query
            audio_id: ID of the audio
            result: Query response
            ttl: Time to live
            
        Returns:
            True if cached
        """
        key = f"query:{audio_id}:{query_hash}"
        return self.set(key, result, ttl)
    
    def clear_audio_cache(self, audio_id: str) -> int:
        """
        Clear all cached data for an audio file.
        
        Args:
            audio_id: ID of the audio
            
        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            return 0
        
        try:
            pattern = self._make_key(f"*:{audio_id}:*")
            keys = list(self.client.scan_iter(match=pattern))
            
            if keys:
                return self.client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.is_connected:
            return {"connected": False}
        
        try:
            info = self.client.info("stats")
            memory = self.client.info("memory")
            
            return {
                "connected": True,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "used_memory_mb": memory.get("used_memory", 0) / 1024 / 1024,
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}
