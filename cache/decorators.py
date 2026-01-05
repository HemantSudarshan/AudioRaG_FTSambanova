"""
Caching Decorators

Easy-to-use decorators for function-level caching.
"""

import logging
import hashlib
import json
from functools import wraps
from typing import Optional, Callable, Any

from cache.redis_cache import RedisCache
from cache.memory_cache import get_memory_cache

logger = logging.getLogger(__name__)

# Global cache instances
_redis_cache: Optional[RedisCache] = None


def init_cache(redis_url: str = "redis://localhost:6379"):
    """Initialize caching with Redis."""
    global _redis_cache
    _redis_cache = RedisCache(url=redis_url)


def _get_cache():
    """Get active cache (Redis or memory fallback)."""
    if _redis_cache and _redis_cache.is_connected:
        return _redis_cache
    return get_memory_cache()


def _make_cache_key(prefix: str, *args, **kwargs) -> str:
    """Create cache key from function arguments."""
    data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    hash_val = hashlib.md5(data.encode()).hexdigest()
    return f"{prefix}:{hash_val}"


def cached(
    prefix: str = "func",
    ttl: int = 3600,
    key_builder: Optional[Callable] = None,
):
    """
    Decorator to cache function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        key_builder: Optional function to build cache key
        
    Example:
        @cached(prefix="my_func", ttl=300)
        def expensive_function(x, y):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = _get_cache()
            
            # Build cache key
            if key_builder:
                key = key_builder(*args, **kwargs)
            else:
                key = _make_cache_key(f"{prefix}:{func.__name__}", *args, **kwargs)
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator


def cache_embeddings(
    ttl: int = 86400,  # 24 hours
):
    """
    Decorator specifically for caching embeddings.
    
    Optimized for embedding functions that take text input.
    
    Args:
        ttl: Time to live in seconds
        
    Example:
        @cache_embeddings()
        def get_embedding(text: str) -> List[float]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(text: str, *args, **kwargs):
            cache = _get_cache()
            
            # Hash the text for key
            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            key = f"embed:{text_hash}"
            
            # Try cache
            result = cache.get(key)
            if result is not None:
                logger.debug(f"Embedding cache hit: {text_hash}")
                return result
            
            # Generate embedding
            result = func(text, *args, **kwargs)
            
            # Cache it
            cache.set(key, result, ttl)
            logger.debug(f"Cached embedding: {text_hash}")
            
            return result
        
        return wrapper
    return decorator


def cache_query_result(
    audio_id: str,
    ttl: int = 3600,
):
    """
    Decorator for caching query results.
    
    Args:
        audio_id: ID of the audio being queried
        ttl: Time to live in seconds
        
    Example:
        @cache_query_result(audio_id="abc123")
        def query_audio(query: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(query: str, *args, **kwargs):
            cache = _get_cache()
            
            # Hash the query
            query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
            key = f"query:{audio_id}:{query_hash}"
            
            # Try cache
            result = cache.get(key)
            if result is not None:
                logger.debug(f"Query cache hit: {query_hash}")
                return result
            
            # Execute query
            result = func(query, *args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl)
            logger.debug(f"Cached query result: {query_hash}")
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(pattern: str):
    """
    Invalidate cache entries matching pattern.
    
    Args:
        pattern: Key pattern to invalidate
    """
    cache = _get_cache()
    
    if isinstance(cache, RedisCache) and cache.is_connected:
        try:
            keys = list(cache.client.scan_iter(match=f"audiorag:{pattern}"))
            if keys:
                cache.client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache keys matching: {pattern}")
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
