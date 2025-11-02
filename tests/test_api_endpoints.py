"""
API Endpoint Tests
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "supported_crops" in data


def test_supported_crops():
    """Test supported crops endpoint"""
    response = client.get("/api/v1/supported-crops")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["crops"]) == 9


def test_disease_detection_without_file():
    """Test disease detection without file"""
    response = client.post("/api/v1/detect-disease")
    assert response.status_code == 422  # Validation error


def test_disease_detection_with_invalid_file():
    """Test disease detection with invalid file"""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/api/v1/detect-disease", files=files)
    assert response.status_code == 400


# Add more tests as needed
