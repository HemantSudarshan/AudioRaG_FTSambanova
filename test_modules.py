"""
AudioRAG Enterprise - Test Script

Verify all enterprise modules are working correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all module imports."""
    print("=" * 50)
    print("Testing Module Imports")
    print("=" * 50)
    
    modules = [
        ("config", "Configuration"),
        ("auth.models", "Auth Models"),
        ("auth.authentication", "Authentication"),
        ("auth.authorization", "Authorization"),
        ("auth.api_keys", "API Keys"),
        ("monitoring.health", "Health Checks"),
        ("audit.logger", "Audit Logger"),
        ("analytics.metrics", "Analytics"),
        ("cache.redis_cache", "Redis Cache"),
        ("cache.memory_cache", "Memory Cache"),
        ("cache.decorators", "Cache Decorators"),
        ("api.routes", "API Routes"),
        ("api.middleware", "API Middleware"),
        ("api.streaming", "Streaming API"),
        ("database.connection", "Database"),
        ("database.models", "DB Models"),
        ("batch.queue", "Job Queue"),
        ("batch.processor", "Batch Processor"),
        ("tenants.models", "Tenant Models"),
        ("tenants.isolation", "Tenant Isolation"),
        ("tenants.billing", "Billing"),
        ("models.domains", "Domain Models"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name} ({module_name})")
            passed += 1
        except Exception as e:
            print(f"  ❌ {display_name} ({module_name}): {e}")
            failed += 1
    
    print(f"\n  Passed: {passed}/{len(modules)}, Failed: {failed}")
    return failed == 0


def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 50)
    print("Testing Configuration")
    print("=" * 50)
    
    try:
        from config import settings, CONFIG
        
        print(f"  App Name: {settings.app_name}")
        print(f"  Version: {settings.app_version}")
        print(f"  Environment: {settings.environment}")
        print(f"  Qdrant URL: {settings.qdrant_url}")
        print(f"  Embed Model: {settings.embed_model_name}")
        print(f"  LLM Model: {settings.llm_model}")
        print(f"  ✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        return False


def test_auth():
    """Test authentication utilities."""
    print("\n" + "=" * 50)
    print("Testing Authentication")
    print("=" * 50)
    
    try:
        from auth.authentication import get_password_hash, verify_password
        from auth.models import RoleType, PermissionType
        
        # Test password hashing
        password = "test_password_123"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed), "Password verification failed"
        print(f"  ✅ Password hashing works")
        
        # Test roles
        print(f"  Available roles: {[r.value for r in RoleType]}")
        print(f"  ✅ Auth models loaded")
        return True
    except Exception as e:
        print(f"  ❌ Auth test failed: {e}")
        return False


def test_cache():
    """Test caching layer."""
    print("\n" + "=" * 50)
    print("Testing Cache Layer")
    print("=" * 50)
    
    try:
        from cache.memory_cache import MemoryCache
        
        cache = MemoryCache(max_size=100)
        
        # Test set/get
        cache.set("test_key", "test_value", ttl=60)
        result = cache.get("test_key")
        assert result == "test_value", "Cache get failed"
        print(f"  ✅ Memory cache works")
        
        # Test stats
        stats = cache.get_stats()
        print(f"  Cache stats: {stats}")
        return True
    except Exception as e:
        print(f"  ❌ Cache test failed: {e}")
        return False


def test_analytics():
    """Test analytics metrics."""
    print("\n" + "=" * 50)
    print("Testing Analytics")
    print("=" * 50)
    
    try:
        from analytics.metrics import MetricsCollector, MetricType
        
        metrics = MetricsCollector()
        
        # Record some metrics
        metrics.record(MetricType.QUERY_COUNT, 1, user_id="test_user")
        metrics.record(MetricType.QUERY_LATENCY, 150.5, user_id="test_user")
        
        # Get summary
        summary = metrics.get_summary(MetricType.QUERY_COUNT, period_hours=1)
        print(f"  Query count: {summary.count}")
        print(f"  ✅ Analytics works")
        return True
    except Exception as e:
        print(f"  ❌ Analytics test failed: {e}")
        return False


def test_health():
    """Test health monitoring."""
    print("\n" + "=" * 50)
    print("Testing Health Monitoring")
    print("=" * 50)
    
    try:
        from monitoring.health import get_liveness, get_readiness
        
        liveness = get_liveness()
        print(f"  Liveness: {liveness}")
        
        readiness = get_readiness()
        print(f"  Readiness: {readiness}")
        print(f"  ✅ Health checks work")
        return True
    except Exception as e:
        print(f"  ❌ Health test failed: {e}")
        return False


def test_domain_models():
    """Test domain-specific models."""
    print("\n" + "=" * 50)
    print("Testing Domain Models")
    print("=" * 50)
    
    try:
        from models.domains import DomainType, DomainModel, DOMAIN_VOCABULARIES
        
        print(f"  Available domains: {[d.value for d in DomainType]}")
        
        # Test healthcare domain
        healthcare = DomainModel(DomainType.HEALTHCARE)
        vocab = healthcare.get_vocabulary()
        print(f"  Healthcare vocabulary: {len(vocab)} terms")
        
        prompt = healthcare.get_prompt("Sample context", "What is the diagnosis?")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  ✅ Domain models work")
        return True
    except Exception as e:
        print(f"  ❌ Domain model test failed: {e}")
        return False


def test_api():
    """Test REST API structure."""
    print("\n" + "=" * 50)
    print("Testing REST API")
    print("=" * 50)
    
    try:
        from api.routes import app, router
        
        print(f"  App title: {app.title}")
        print(f"  API routes registered")
        print(f"  Docs URL: {app.docs_url}")
        print(f"  ✅ API structure valid")
        return True
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  AudioRAG Enterprise - Module Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Auth", test_auth),
        ("Cache", test_cache),
        ("Analytics", test_analytics),
        ("Health", test_health),
        ("Domains", test_domain_models),
        ("API", test_api),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            results.append((name, test_func()))
        except Exception as e:
            print(f"  ❌ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
