"""
In-Memory Cache Implementation

LRU cache fallback when Redis is unavailable.
"""

import logging
import hashlib
import time
from typing import Optional, Any, Dict, Tuple
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)


class MemoryCache:
    """
    Thread-safe LRU cache with TTL support.
    
    Used as fallback when Redis is unavailable.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of items
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = Lock()
        
        logger.info(f"Memory cache initialized (max_size={max_size})")
    
    def _is_expired(self, expiry: float) -> bool:
        """Check if entry is expired."""
        return time.time() > expiry
    
    def _evict_expired(self):
        """Remove expired entries."""
        now = time.time()
        expired = [k for k, (_, exp) in self._cache.items() if now > exp]
        for key in expired:
            del self._cache[key]
    
    def _evict_oldest(self):
        """Remove oldest entries if over capacity."""
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self._lock:
            if key not in self._cache:
                logger.debug(f"Memory cache miss: {key}")
                return None
            
            value, expiry = self._cache[key]
            
            if self._is_expired(expiry):
                del self._cache[key]
                logger.debug(f"Memory cache expired: {key}")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            logger.debug(f"Memory cache hit: {key}")
            return value
    
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
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        with self._lock:
            self._evict_expired()
            self._evict_oldest()
            
            self._cache[key] = (value, expiry)
            self._cache.move_to_end(key)
            
        logger.debug(f"Memory cache set: {key} (ttl={ttl}s)")
        return True
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Memory cache delete: {key}")
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                return False
            
            _, expiry = self._cache[key]
            return not self._is_expired(expiry)
    
    def clear(self):
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
        logger.info("Memory cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            self._evict_expired()
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size * 100,
            }


# ===================================
# Global Instance
# ===================================

_memory_cache: Optional[MemoryCache] = None


def get_memory_cache() -> MemoryCache:
    """Get global memory cache instance."""
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = MemoryCache()
    return _memory_cache
