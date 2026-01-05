"""
AudioRAG - Quick Validation Script

Validate code syntax without requiring all dependencies.
"""

import os
import sys
import py_compile
from pathlib import Path

def validate_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """Validate all Python files."""
    print("🔍 AudioRAG - Syntax Validation")
    print("=" * 50)
    
    root = Path(__file__).parent
    python_files = list(root.rglob("*.py"))
    
    # Exclude venv
    python_files = [f for f in python_files if "venv" not in str(f) and "__pycache__" not in str(f)]
    
    passed = 0
    failed = 0
    errors = []
    
    for file_path in python_files:
        rel_path = file_path.relative_to(root)
        valid, error = validate_syntax(file_path)
        
        if valid:
            passed += 1
        else:
            failed += 1
            errors.append((rel_path, error))
    
    print(f"\n✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if errors:
        print("\nErrors found:")
        for path, error in errors:
            print(f"\n  {path}:")
            print(f"    {error}")
        return False
    else:
        print("\n🎉 All files have valid syntax!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
