"""
End-to-end tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_complete_query_workflow():
    """Test complete user query workflow."""
    # Health check first
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
    # Submit query
    query_payload = {
        "question": "What is the grading policy?",
        "top_k": 3
    }
    
    response = client.post("/api/query", json=query_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0


def test_query_with_course_filter():
    """Test query with course filtering."""
    query_payload = {
        "question": "What are the prerequisites?",
        "course": "Introduction_to_Computing",
        "top_k": 3
    }
    
    response = client.post("/api/query", json=query_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data


def test_conversation_context():
    """Test multi-turn conversation maintains context."""
    session_id = "test-session-123"
    
    # First query
    response1 = client.post("/api/query", json={
        "question": "What is the final exam percentage?",
        "session_id": session_id
    })
    
    assert response1.status_code == 200
    
    # Follow-up query
    response2 = client.post("/api/query", json={
        "question": "And the midterm?",
        "session_id": session_id
    })
    
    assert response2.status_code == 200
    # Should successfully process follow-up question


def test_query_validation_missing_question():
    """Test query validation with missing question."""
    query_payload = {
        "top_k": 3
    }
    
    response = client.post("/api/query", json=query_payload)
    
    # Should return validation error
    assert response.status_code == 422


def test_query_validation_invalid_top_k():
    """Test query validation with invalid top_k."""
    query_payload = {
        "question": "Test question",
        "top_k": 100  # Exceeds maximum
    }
    
    response = client.post("/api/query", json=query_payload)
    
    # Should return validation error
    assert response.status_code == 422


def test_streaming_endpoint():
    """Test streaming endpoint returns SSE."""
    query_payload = {
        "question": "What is the grading policy?",
        "top_k": 3
    }
    
    response = client.post("/api/query/stream", json=query_payload)
    
    # Should return 200 for streaming
    assert response.status_code == 200
    # Content type should be event-stream
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_new_conversation_flag():
    """Test starting new conversation."""
    session_id = "test-session-456"
    
    # First query
    response1 = client.post("/api/query", json={
        "question": "First question",
        "session_id": session_id
    })
    
    assert response1.status_code == 200
    
    # Start new conversation
    response2 = client.post("/api/query", json={
        "question": "New conversation question",
        "session_id": session_id,
        "start_new_conversation": True
    })
    
    assert response2.status_code == 200
    # Should process successfully with new context
