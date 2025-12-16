#!/usr/bin/env python3
"""
Validate evaluation setup before running.

Checks:
1. test_queries.json is valid JSON
2. Required dependencies are available
3. Paths are correct
4. Scripts can be imported
"""

import json
import sys
from pathlib import Path

def validate_json_file(file_path: Path) -> bool:
    """Validate JSON file syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ {file_path} is valid JSON")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ {file_path} has JSON syntax error: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"✗ {file_path} not found", file=sys.stderr)
        return False

def validate_test_queries(file_path: Path) -> bool:
    """Validate test queries structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        if not isinstance(queries, list):
            print(f"✗ {file_path} should contain a JSON array", file=sys.stderr)
            return False
        
        if len(queries) == 0:
            print(f"✗ {file_path} is empty", file=sys.stderr)
            return False
        
        required_fields = ['query']
        for i, query in enumerate(queries):
            for field in required_fields:
                if field not in query:
                    print(f"✗ Query {i+1} missing required field: {field}", file=sys.stderr)
                    return False
        
        queries_with_ref = sum(1 for q in queries if q.get('reference_answer'))
        queries_without_ref = len(queries) - queries_with_ref
        
        print(f"✓ {file_path} has {len(queries)} queries")
        print(f"  - {queries_with_ref} with reference_answer")
        print(f"  - {queries_without_ref} without reference_answer")
        return True
    except Exception as e:
        print(f"✗ Error validating {file_path}: {e}", file=sys.stderr)
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    dependencies = {
        'rouge_score': 'rouge-score',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed (pip install {package})", file=sys.stderr)
            missing.append(package)
    
    return len(missing) == 0

def check_paths():
    """Check if required paths exist."""
    base_path = Path(__file__).resolve().parents[1]
    
    required_paths = {
        'scripts/test_queries.json': 'Test queries file',
        'scripts/evaluate_rag_system.py': 'Evaluation script',
        'scripts/create_visualizations.py': 'Visualization script',
        'scripts/fill_results_tables.py': 'Fill tables script',
        'overleaf-report/AI_PROJECT_REPORT/experiements/results.tex': 'Results LaTeX file',
    }
    
    all_exist = True
    for rel_path, description in required_paths.items():
        full_path = base_path / rel_path
        if full_path.exists():
            print(f"✓ {description}: {rel_path}")
        else:
            print(f"✗ {description} NOT found: {rel_path}", file=sys.stderr)
            all_exist = False
    
    return all_exist

def main():
    print("=" * 60)
    print("EVALUATION SETUP VALIDATION")
    print("=" * 60)
    
    base_path = Path(__file__).resolve().parents[1]
    test_queries = base_path / "scripts" / "test_queries.json"
    
    print("\n1. Validating JSON files...")
    json_ok = validate_json_file(test_queries)
    
    print("\n2. Validating test queries structure...")
    queries_ok = validate_test_queries(test_queries) if json_ok else False
    
    print("\n3. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n4. Checking file paths...")
    paths_ok = check_paths()
    
    print("\n" + "=" * 60)
    if json_ok and queries_ok and deps_ok and paths_ok:
        print("✓ All checks passed! Evaluation setup is ready.")
        print("\nNext steps:")
        print("1. Run evaluation: python3 scripts/evaluate_rag_system.py")
        print("2. Generate visualizations: python3 scripts/create_visualizations.py --results scripts/evaluation_results.json")
        print("3. Fill LaTeX tables: python3 scripts/fill_results_tables.py --results scripts/evaluation_results.json")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

