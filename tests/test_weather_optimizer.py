"""
ShutterIQ - Weather Optimizer Tests

This suite runs checks on the weather and solar scheduling optimizer,
asserting golden hour boundaries, slot scoring mechanics under various weather
conditions (including a rain veto), and graceful degradation during API timeouts or failures.
"""

import pytest
from datetime import date, datetime
import zoneinfo
from unittest.mock import patch, MagicMock
import requests
import numpy as np

from models.weather_optimizer import (
    get_solar_windows,
    calculate_lighting_score,
    calculate_cloud_score,
    score_slots
)

# Coordinates for Colombo, Sri Lanka
LAT = 6.9271
LON = 79.8612
TZ = "Asia/Colombo"

# Mock Open-Meteo response
MOCK_WEATHER_RESPONSE = {
    "hourly": {
        "time": [
            "2026-07-26T07:00",
            "2026-07-26T08:00",
            "2026-07-26T09:00",
            "2026-07-26T10:00",
            "2026-07-26T11:00",
            "2026-07-26T12:00",
            "2026-07-26T13:00",
            "2026-07-26T14:00",
            "2026-07-26T15:00",
            "2026-07-26T16:00",
            "2026-07-26T17:00",
            "2026-07-26T18:00"
        ],
        "temperature_2m": [28.0, 28.5, 29.0, 29.5, 30.0, 25.0, 29.8, 29.5, 29.0, 28.5, 28.0, 27.5],
        "precipitation_probability": [0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0],  # 100% rain at 12:00
        "cloud_cover": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]  # Perfect 30% cloud cover everywhere
    }
}


def test_golden_hour_astronomical_calculation():
    """
    Verifies that astral computes sunrise, sunset, and golden hour windows
    deterministically for a known date and location without hitches.
    """
    target_date = date(2026, 7, 26)
    windows = get_solar_windows(LAT, LON, TZ, target_date)

    # 1. Assert windows are resolved
    assert windows["morning_golden"] is not None
    assert windows["evening_golden"] is not None
    assert windows["morning_blue"] is not None
    assert windows["evening_blue"] is not None

    # 2. Check chronological ordering
    for k in ["morning_golden", "evening_golden", "morning_blue", "evening_blue"]:
        start, end = windows[k]
        assert start < end, f"Window {k} has start time after end time."
        assert start.tzinfo is not None, f"Window {k} is timezone-naive."


def test_cloud_score_curve():
    """
    Sanity checks the cloud cover rating system.
    Diffused (30%) should be 1.0, clear sky (0%) should be 0.6, overcast (100%) should be 0.4.
    """
    assert calculate_cloud_score(30.0) == pytest.approx(1.0)
    assert calculate_cloud_score(0.0) == pytest.approx(0.6)
    assert calculate_cloud_score(100.0) == pytest.approx(0.4)
    assert calculate_cloud_score(15.0) == pytest.approx(0.8)
    assert calculate_cloud_score(65.0) == pytest.approx(0.7)


@patch("models.weather_optimizer.requests.get")
def test_slot_scoring_under_perfect_conditions(mock_get):
    """
    Tests slot scoring under mock perfect weather.
    Verifies that golden hour slots with zero rain and ideal cloud cover score highest.
    """
    # Configure mock response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_WEATHER_RESPONSE
    mock_get.return_value = mock_resp

    target_date = date(2026, 7, 26)
    results = score_slots(LAT, LON, TZ, target_date, target_date)

    assert len(results) == 11, "Expected 11 daylight hourly slots (7 AM to 5 PM start times)"
    
    # The top slot should correspond to the evening golden hour (17:00 to 18:00)
    top_slot = results[0]
    assert "17:00:00" in top_slot["start_time"]
    assert top_slot["precip_score"] == 1.0, "Zero precip must give 1.0 precip score"
    assert top_slot["cloud_score"] == 1.0, "30% cloud cover must give 1.0 cloud score"
    assert top_slot["score"] > 0.8, f"Top slot score {top_slot['score']} should be high"
    assert top_slot["weather_available"] is True


@patch("models.weather_optimizer.requests.get")
def test_slot_scoring_veto_on_heavy_rain(mock_get):
    """
    Asserts that precipitation functions as a multiplicative veto.
    If rain probability is 100%, the slot score must be 0.0 regardless of light quality.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_WEATHER_RESPONSE
    mock_get.return_value = mock_resp

    target_date = date(2026, 7, 26)
    results = score_slots(LAT, LON, TZ, target_date, target_date)

    # Locate the 12:00 (midday) slot where we forced 100% precip
    midday_slots = [s for s in results if "12:00:00" in s["start_time"]]
    assert len(midday_slots) == 1
    midday_slot = midday_slots[0]

    assert midday_slot["precip_score"] == 0.0, "100% rain probability must yield 0.0 precip score"
    assert midday_slot["score"] == 0.0, "Multiplicative rain veto failed: slot score should be 0.0"


@patch("models.weather_optimizer.requests.get")
def test_graceful_degradation_on_api_failure(mock_get):
    """
    Verifies that the optimizer degradades gracefully to solar-only scoring
    if Open-Meteo times out or fails (e.g. returns 500).
    """
    # Mock a timeout exception
    mock_get.side_effect = requests.exceptions.Timeout("API Timeout")

    target_date = date(2026, 7, 26)
    
    # Assert that no exceptions are raised and slot scoring completes
    try:
        results = score_slots(LAT, LON, TZ, target_date, target_date)
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"score_slots crashed on API timeout: {e}")
        
    assert success
    assert len(results) == 11
    
    # Assert weather attributes degrade properly
    for slot in results:
        assert slot["weather_available"] is False
        assert np.isnan(slot["cloud_cover_percent"])
        assert np.isnan(slot["precip_prob_percent"])
        assert np.isnan(slot["temperature_c"])
        assert slot["cloud_score"] == 1.0, "Fallback cloud score must be 1.0"
        assert slot["precip_score"] == 1.0, "Fallback precip score must be 1.0"
        # Total score must equal the lighting score under degradation
        assert slot["score"] == slot["lighting_score"]

    # Golden hour slots should still rank highest
    # Colombo evening golden hour is 17:00 slot
    assert "17:00:00" in results[0]["start_time"]
