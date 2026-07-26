"""
ShutterIQ - Dynamic Pricing Model Tests

This suite runs sanity checks on the trained baseline and elasticity models,
asserting training correctness, price optimization bounds, expected revenue maximization behavior,
and willingness-to-pay recovery performance.
"""

import os
import pytest
import pandas as pd
import numpy as np
from models.pricing_model import (
    train_models,
    _get_artifacts,
    recommend_price_and_expected_revenue,
    calculate_implied_wtp
)

def test_pricing_model_training_execution():
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


def test_pricing_model_artifacts_loading():
    """
    Checks that the model pickle file loads correctly and contains all necessary components.
    """
    artifacts = _get_artifacts()
    
    assert "elasticity_pipeline" in artifacts, "Elasticity pipeline missing from model artifacts"
    assert "baseline_pipeline" in artifacts, "Baseline regressor pipeline missing from model artifacts"
    assert "metrics" in artifacts, "Performance metrics missing from model artifacts"
    
    metrics = artifacts["metrics"]
    assert "baseline" in metrics
    assert "elasticity" in metrics
    assert "wtp_recovery" in metrics


def test_recommended_prices_bounds_and_ordering():
    """
    Asserts that recommended prices are positive, fall within realistic constraints,
    and preserve shoot-type hierarchies (weddings cost more than portraits).
    """
    # Portrait booking request
    portrait_features = {
        "shoot_type": "portrait",
        "season": "regular_season",
        "day_of_week": "Monday",
        "lead_time_days": 30,
        "photographer_experience_years": 5,
        "num_people": 2
    }
    
    rec_p, exp_rev_p = recommend_price_and_expected_revenue(portrait_features)
    assert rec_p > 0, "Portrait recommended price must be positive"
    assert 50.0 <= rec_p <= 400.0, f"Portrait price {rec_p} is outside plausible bounds (50-400)"
    assert exp_rev_p > 0, "Expected revenue must be positive"
    
    # Wedding booking request
    wedding_features = {
        "shoot_type": "wedding",
        "season": "wedding_season",
        "day_of_week": "Saturday",
        "lead_time_days": 90,
        "photographer_experience_years": 10,
        "num_people": 150
    }
    
    rec_w, exp_rev_w = recommend_price_and_expected_revenue(wedding_features)
    assert rec_w > 0, "Wedding recommended price must be positive"
    assert 1000.0 <= rec_w <= 5000.0, f"Wedding price {rec_w} is outside plausible bounds (1000-5000)"
    assert exp_rev_w > 0, "Expected revenue must be positive"
    
    # Check ordering
    assert rec_w > rec_p, f"Hierarchy error: Wedding price {rec_w} should exceed Portrait price {rec_p}"


def test_expected_revenue_optimization_picks_sensible_price():
    """
    Verifies that the expected revenue optimization does not always pick the extreme
    ends of the search grid. If a price is too low, revenue is lost; if too high,
    acceptance drops. The optimized price should be interior to the search boundaries.
    """
    artifacts = _get_artifacts()
    elasticity_pipeline = artifacts["elasticity_pipeline"]
    baseline_pipeline = artifacts["baseline_pipeline"]
    from models.pricing_model import optimize_price
    
    booking_features = {
        "shoot_type": "portrait",
        "season": "regular_season",
        "day_of_week": "Monday",
        "lead_time_days": 30,
        "photographer_experience_years": 5,
        "num_people": 2
    }
    
    # Let's inspect the candidate search bounds
    input_df = pd.DataFrame([booking_features])
    pred_baseline_log = float(baseline_pipeline.predict(input_df[["shoot_type", "season", "day_of_week", "lead_time_days", "photographer_experience_years", "num_people"]])[0])
    pred_baseline = np.exp(pred_baseline_log)
    
    low_bound = max(10.0, 0.5 * pred_baseline)
    high_bound = 2.0 * pred_baseline
    
    rec_price = optimize_price(elasticity_pipeline, baseline_pipeline, booking_features)
    
    # Assert that the chosen price is strictly within the interior of the search space,
    # not hard-clamped at the bounds.
    assert low_bound < rec_price < high_bound, (
        f"Optimization failed: recommended price {rec_price} is clamped at the "
        f"grid boundaries [{low_bound}, {high_bound}]"
    )


def test_wtp_recovery_accuracy():
    """
    Asserts that the willingness-to-pay recovery metrics stored during training
    satisfy minimum accuracy requirements: overall correlation > 0.90 and overall R² > 0.85.
    """
    artifacts = _get_artifacts()
    wtp_metrics = artifacts["metrics"]["wtp_recovery"]
    
    corr = wtp_metrics["overall_corr"]
    r2 = wtp_metrics["overall_r2"]
    
    assert corr >= 0.90, f"Willingness-to-pay recovery correlation ({corr:.4f}) is lower than threshold (0.90)"
    assert r2 >= 0.85, f"Willingness-to-pay recovery R² ({r2:.4f}) is lower than threshold (0.85)"
