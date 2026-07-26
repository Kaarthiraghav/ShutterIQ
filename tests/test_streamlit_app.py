"""
ShutterIQ App - Streamlit Helpers Tests

Verifies formatting logic, weather badge categories, and single-row SHAP explanation
calculations extracted from the Streamlit UI to ensure high test coverage.
"""

import pytest
import os
from app.helpers import (
    format_duration,
    get_weather_badge,
    explain_duration_factors
)


def test_format_duration():
    """
    Asserts duration formatting maps raw minutes into user-friendly strings.
    """
    assert format_duration(45.0) == "45 mins"
    assert format_duration(60.0) == "1 hr"
    assert format_duration(75.3) == "1 hr 15 mins"
    assert format_duration(120.0) == "2 hrs"
    assert format_duration(150.0) == "2 hrs 30 mins"
    assert format_duration(360.0) == "6 hrs"


def test_get_weather_badge():
    """
    Asserts weather badge ranges map correctly to quality categories.
    """
    assert get_weather_badge(0.95) == "🌟 Excellent"
    assert get_weather_badge(0.80) == "🌟 Excellent"
    assert get_weather_badge(0.70) == "✨ Good"
    assert get_weather_badge(0.60) == "✨ Good"
    assert get_weather_badge(0.50) == "⛅ Fair"
    assert get_weather_badge(0.40) == "⛅ Fair"
    assert get_weather_badge(0.30) == "🌧️ Poor"
    assert get_weather_badge(0.0) == "🌧️ Poor"


def test_explain_duration_factors():
    """
    Sanity checks duration model single-row SHAP attributions.
    Asserts base values and factors are correctly extracted and structured.
    """
    # Use a basic portrait input
    res = explain_duration_factors(
        shoot_type="portrait",
        num_people=2,
        location_type="indoor",
        lighting_setup="studio_lights",
        photographer_experience_years=5
    )

    assert "base_value" in res
    assert "prediction" in res
    assert "factors" in res

    assert res["prediction"] > 0
    assert res["base_value"] > 200.0  # Dataset mean is around ~249.12 mins

    factors = res["factors"]
    expected_keys = {
        "Shoot Type (Service)",
        "Location Setting",
        "Lighting Setup",
        "Group Size (People)",
        "Photographer Experience"
    }
    assert set(factors.keys()) == expected_keys

    # Portrait should have a large negative shoot type offset
    assert factors["Shoot Type (Service)"] < 0.0
