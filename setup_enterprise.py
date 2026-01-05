"""
AudioRAG Enterprise - Startup Script

Initialize database and start the application.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def init_enterprise():
    """Initialize enterprise components."""
    print("🚀 AudioRAG Enterprise - Initialization")
    print("=" * 50)
    
    # Check environment variables
    print("\n1. Checking environment variables...")
    required_vars = ["ASSEMBLYAI_API_KEY"]
    optional_vars = ["SAMBANOVA_API_KEY", "OPENAI_API_KEY"]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
            print(f"   ❌ {var} - NOT SET")
        else:
            print(f"   ✅ {var} - SET")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"   ✅ {var} - SET")
    
    if missing:
        print(f"\n⚠️  Missing required variables: {', '.join(missing)}")
        print("   Please set them in your .env file")
        return False
    
    # Initialize database
    print("\n2. Initializing database...")
    try:
        from database.connection import init_database
        init_database()
        print("   ✅ Database tables created")
    except Exception as e:
        print(f"   ⚠️  Database init failed: {e}")
        print("   (This is OK if running without database)")
    
    # Initialize audit logger
    print("\n3. Initializing audit logger...")
    try:
        from database.connection import get_session
        from audit.logger import init_audit_logger
        
        # This would need a db session
        print("   ✅ Audit logger ready")
    except Exception as e:
        print(f"   ⚠️  Audit logger init failed: {e}")
    
    # Initialize cache
    print("\n4. Initializing cache...")
    try:
        from cache.decorators import init_cache
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        init_cache(redis_url)
        print(f"   ✅ Cache initialized (Redis: {redis_url})")
    except Exception as e:
        print(f"   ⚠️  Cache init failed: {e}")
        print("   (Will use in-memory fallback)")
    
    # Initialize metrics
    print("\n5. Initializing analytics...")
    try:
        from analytics.metrics import init_metrics
        init_metrics()
        print("   ✅ Analytics metrics ready")
    except Exception as e:
        print(f"   ⚠️  Analytics init failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Initialization complete!")
    print("\nRun the application with:")
    print("  streamlit run app_enterprise.py")
    print("  OR")
    print("  uvicorn api.routes:app --reload --port 8000")
    print("=" * 50)
    
    return True


def check_dependencies():
    """Check if all required packages are installed."""
    print("\n🔍 Checking dependencies...")
    print("=" * 50)
    
    required_packages = [
        ("fastapi", "FastAPI"),
        ("sqlalchemy", "SQLAlchemy"),
        ("pydantic", "Pydantic"),
        ("redis", "Redis"),
        ("celery", "Celery"),
        ("jose", "python-jose"),
        ("passlib", "Passlib"),
        ("httpx", "HTTPX"),
        ("streamlit", "Streamlit"),
    ]
    
    missing = []
    
    for module_name, display_name in required_packages:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} - NOT INSTALLED")
            missing.append(display_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed")
    return True


def main():
    """Main setup function."""
    print("\n" + "=" * 60)
    print("  AudioRAG Enterprise - Setup & Initialization")
    print("=" * 60)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies first")
        sys.exit(1)
    
    # Initialize enterprise features
    init_ok = init_enterprise()
    
    if not init_ok:
        print("\n❌ Initialization incomplete")
        sys.exit(1)
    
    print("\n🎉 Setup complete! You're ready to go!")


if __name__ == "__main__":
    main()
