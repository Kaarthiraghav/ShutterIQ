"""
ShutterIQ - Shoot Duration Model Tests

This suite runs sanity checks on the trained XGBoost duration model,
asserting prediction validity, positive outputs, range bounds, and buffer behaviors.
"""

import os
import pytest
import pandas as pd
import numpy as np
from shutteriq.models.duration_model import (
    predict_duration_with_buffer,
    _get_artifacts,
    train_models
)

def test_model_training_execution():
    """
    Checks that the train_models function executes successfully and updates artifacts.
    """
    try:
        train_models()
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"train_models failed with exception: {e}")
    assert success


def test_model_artifacts_loading():
    """
    Checks that the model pickle file loads correctly and contains all necessary components.
    """
    artifacts = _get_artifacts()
    
    assert "xgb_pipeline" in artifacts, "XGBoost pipeline missing from model artifacts"
    assert "lr_pipeline" in artifacts, "Baseline Linear Regression pipeline missing from model artifacts"
    assert "buffers" in artifacts, "Scheduling buffers missing from model artifacts"
    assert "overall_buffer" in artifacts, "Overall buffer missing from model artifacts"
    assert "metrics" in artifacts, "Performance metrics missing from model artifacts"


def test_predictions_positive_and_plausible():
    """
    Asserts that predictions and buffers are positive and fall within realistic hourly constraints.
    """
    # 1. Portrait Shoot prediction sanity checks
    pred_p, buf_p = predict_duration_with_buffer(
        shoot_type="portrait",
        num_people=2,
        location_type="indoor",
        lighting_setup="studio_lights",
        photographer_experience_years=5
    )
    assert pred_p > 0, "Portrait prediction should be positive"
    assert 15.0 <= pred_p <= 120.0, f"Portrait predicted duration {pred_p} mins is outside plausible bounds (15-120 mins)"
    assert buf_p > 0, "Portrait buffer must be positive"
    
    # 2. Wedding Shoot prediction sanity checks (highly complex, many people)
    pred_w, buf_w = predict_duration_with_buffer(
        shoot_type="wedding",
        num_people=150,
        location_type="outdoor",
        lighting_setup="natural_light",
        photographer_experience_years=10
    )
    assert pred_w > 0, "Wedding prediction should be positive"
    assert 180.0 <= pred_w <= 1200.0, f"Wedding predicted duration {pred_w} mins is outside plausible bounds (180-1200 mins)"
    assert buf_w > 0, "Wedding buffer must be positive"


def test_buffer_scaling_by_type():
    """
    Validates that the shoot-type specific buffers scale proportionally to risk (Option A).
    Weddings (high complexity/variance) must have a larger buffer than portraits.
    """
    _, buf_p = predict_duration_with_buffer(
        shoot_type="portrait",
        num_people=1,
        location_type="indoor",
        lighting_setup="none",
        photographer_experience_years=2
    )
    
    _, buf_w = predict_duration_with_buffer(
        shoot_type="wedding",
        num_people=100,
        location_type="outdoor",
        lighting_setup="natural_light",
        photographer_experience_years=10
    )
    
    assert buf_w > buf_p, (
        f"Variance-based scaling error: Wedding buffer ({buf_w} mins) "
        f"should exceed Portrait buffer ({buf_p} mins)."
    )
