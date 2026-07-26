"""
ShutterIQ - FastAPI Integration Tests

Sanity checks all FastAPI endpoints (GET /health, POST /predict-duration,
POST /recommend-price, POST /best-slots, and POST /schedule-suggestion)
using FastAPI's TestClient.

Also tests invalid input schema boundaries and asserts correct validation behavior.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add parent directory to sys.path to allow clean imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.main import app

client = TestClient(app)

# Mock Open-Meteo response matching expected structure
MOCK_WEATHER_RESPONSE = {
    "hourly": {
        "time": [
            "2026-07-26T07:00", "2026-07-26T08:00", "2026-07-26T09:00",
            "2026-07-26T10:00", "2026-07-26T11:00", "2026-07-26T12:00",
            "2026-07-26T13:00", "2026-07-26T14:00", "2026-07-26T15:00",
            "2026-07-26T16:00", "2026-07-26T17:00", "2026-07-26T18:00"
        ],
        "temperature_2m": [25.0] * 12,
        "precipitation_probability": [0] * 12,
        "cloud_cover": [30] * 12
    }
}


def test_health_check_endpoint():
    """
    Asserts GET /health returns HTTP 200 OK.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_predict_duration_endpoint_success():
    """
    Verifies POST /predict-duration succeeds with a valid payload.
    """
    payload = {
        "shoot_type": "portrait",
        "num_people": 2,
        "location_type": "indoor",
        "lighting_setup": "studio_lights",
        "photographer_experience_years": 5
    }
    response = client.post("/predict-duration", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_duration_minutes" in data
    assert "recommended_buffer_minutes" in data
    assert data["predicted_duration_minutes"] > 0
    assert data["recommended_buffer_minutes"] > 0


def test_predict_duration_endpoint_validation_errors():
    """
    Asserts POST /predict-duration raises validation error on invalid input.
    """
    # 1. Invalid category
    payload_bad_cat = {
        "shoot_type": "invalid_shoot",
        "num_people": 2,
        "location_type": "indoor",
        "lighting_setup": "studio_lights",
        "photographer_experience_years": 5
    }
    response = client.post("/predict-duration", json=payload_bad_cat)
    assert response.status_code == 422

    # 2. Out of range experience
    payload_bad_exp = {
        "shoot_type": "portrait",
        "num_people": 2,
        "location_type": "indoor",
        "lighting_setup": "studio_lights",
        "photographer_experience_years": 25  # Max is 20
    }
    response = client.post("/predict-duration", json=payload_bad_exp)
    assert response.status_code == 422


def test_recommend_price_endpoint_success():
    """
    Verifies POST /recommend-price succeeds with a valid payload.
    """
    payload = {
        "shoot_type": "portrait",
        "season": "regular_season",
        "day_of_week": "Monday",
        "lead_time_days": 30,
        "photographer_experience_years": 5,
        "num_people": 2
    }
    response = client.post("/recommend-price", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_price" in data
    assert "expected_revenue" in data
    assert data["recommended_price"] > 0
    assert data["expected_revenue"] > 0


def test_recommend_price_endpoint_validation_errors():
    """
    Asserts POST /recommend-price raises validation error on out-of-bounds parameters.
    """
    # Negative lead time
    payload = {
        "shoot_type": "portrait",
        "season": "regular_season",
        "day_of_week": "Monday",
        "lead_time_days": -5,
        "photographer_experience_years": 5,
        "num_people": 2
    }
    response = client.post("/recommend-price", json=payload)
    assert response.status_code == 422


@patch("models.weather_optimizer.requests.get")
def test_best_slots_endpoint_success(mock_get):
    """
    Verifies POST /best-slots returns optimal scheduling slots when weather API is mocked.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_WEATHER_RESPONSE
    mock_get.return_value = mock_resp

    payload = {
        "latitude": 6.9271,
        "longitude": 79.8612,
        "start_date": "2026-07-26",
        "end_date": "2026-07-26",
        "timezone": "Asia/Colombo"
    }
    response = client.post("/best-slots", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "slots" in data
    assert len(data["slots"]) > 0
    
    first_slot = data["slots"][0]
    assert "start_time" in first_slot
    assert "score" in first_slot
    assert first_slot["weather_available"] is True


def test_best_slots_endpoint_validation_errors():
    """
    Asserts POST /best-slots raises validation errors on out-of-bounds coordinates.
    """
    # Invalid latitude > 90
    payload = {
        "latitude": 120.0,
        "longitude": 79.8612,
        "start_date": "2026-07-26",
        "end_date": "2026-07-26",
        "timezone": "Asia/Colombo"
    }
    response = client.post("/best-slots", json=payload)
    assert response.status_code == 422


@patch("models.weather_optimizer.requests.get")
def test_schedule_suggestion_endpoint_outdoor(mock_get):
    """
    Verifies POST /schedule-suggestion combined endpoint outputs correct results
    for an outdoor shoot, including fetching slot score recommendations.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_WEATHER_RESPONSE
    mock_get.return_value = mock_resp

    payload = {
        "shoot_type": "portrait",
        "num_people": 2,
        "location_type": "outdoor",
        "lighting_setup": "natural_light",
        "photographer_experience_years": 5,
        "season": "regular_season",
        "day_of_week": "Saturday",
        "lead_time_days": 10,
        "latitude": 6.9271,
        "longitude": 79.8612,
        "start_date": "2026-07-26",
        "end_date": "2026-07-26",
        "timezone": "Asia/Colombo"
    }
    response = client.post("/schedule-suggestion", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "duration_prediction" in data
    assert "price_recommendation" in data
    assert "best_slots" in data
    
    assert len(data["best_slots"]) > 0
    assert data["duration_prediction"]["predicted_duration_minutes"] > 0
    assert data["price_recommendation"]["recommended_price"] > 0


def test_schedule_suggestion_endpoint_indoor():
    """
    Verifies POST /schedule-suggestion skips slot/weather scoring
    when location_type is 'indoor', returning an empty slot list.
    """
    payload = {
        "shoot_type": "portrait",
        "num_people": 2,
        "location_type": "indoor",
        "lighting_setup": "studio_lights",
        "photographer_experience_years": 5,
        "season": "regular_season",
        "day_of_week": "Saturday",
        "lead_time_days": 10,
        "latitude": 6.9271,
        "longitude": 79.8612,
        "start_date": "2026-07-26",
        "end_date": "2026-07-26",
        "timezone": "Asia/Colombo"
    }
    # We do NOT patch requests.get here; if it tried to make a request,
    # it would either fail or hit the network. By asserting slots are empty
    # without patching, we confirm the API doesn't query weather for indoor slots.
    response = client.post("/schedule-suggestion", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "duration_prediction" in data
    assert "price_recommendation" in data
    assert "best_slots" in data
    assert data["best_slots"] == []
