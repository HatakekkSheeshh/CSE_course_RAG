# Testing Directory

This directory contains the test suite for the RAG system.

## Structure

```
testing/
├── conftest.py              # pytest configuration and shared fixtures
├── unit/                    # Unit tests (70% of tests)
│   ├── test_embedding.py    # Embedding module tests
│   ├── test_indexing.py     # Indexing module tests
│   └── test_reranker.py     # Reranker module tests
├── integration/             # Integration tests (20% of tests)
│   └── test_retrieval_pipeline.py  # Retrieval pipeline integration tests
└── e2e/                     # End-to-end tests (10% of tests)
    └── test_api_endpoints.py       # API endpoint E2E tests
```

## Running Tests

### Run all tests
```bash
pytest testing/
```

### Run with coverage
```bash
pytest testing/ --cov=. --cov-report=html --cov-report=term
```

### Run specific test category
```bash
# Unit tests only
pytest testing/unit/

# Integration tests only
pytest testing/integration/

# E2E tests only
pytest testing/e2e/
```

### Run specific test file
```bash
pytest testing/unit/test_embedding.py
```

### Run specific test function
```bash
pytest testing/unit/test_embedding.py::test_embedding_initialization
```

## Test Coverage Goals

- **Overall Coverage**: ≥80%
- **Core Modules**: ≥90% (embedding, indexing, retrieval)
- **API Endpoints**: 100%
- **Critical Paths**: 100% (query pipeline, answer generation)

## Prerequisites

Before running tests:
1. Install dependencies: `pip install -r requirements.txt`
2. Install test dependencies: `pip install pytest pytest-cov`
3. Ensure FAISS indices are built: `python -m rag.index_builder`

## Notes

- Integration tests require actual FAISS indices to be present
- E2E tests require the FastAPI application to be importable
- Some tests may be skipped if required data is not available
