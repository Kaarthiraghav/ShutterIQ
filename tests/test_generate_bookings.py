"""
ShutterIQ - Synthetic Booking Generator Tests

This test suite performs sanity checks on the synthetic data generation engine,
including shape, typing, boundaries, pricing scaling, and price elasticity properties.
"""

import pytest
import pandas as pd
import numpy as np
from shutteriq.simulate.generate_bookings import (
    generate_dataset,
    calculate_duration,
    calculate_price_and_acceptance
)

def test_dataset_generation_shapes_and_types():
    """
    Verifies that the generated dataset matches shapes, columns, and types.
    """
    df = generate_dataset(num_records=100, seed=123)
    
    # 1. Shape check
    assert len(df) == 100
    
    # 2. Check all required columns exist
    required_cols = [
        "booking_id", "shoot_type", "num_people", "location_type",
        "lighting_setup", "photographer_experience_years", "lead_time_days",
        "day_of_week", "season", "requested_datetime", "actual_duration_minutes",
        "quoted_price", "true_wtp", "booking_accepted"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col}"
        
    # 3. Check for null values
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"
    
    # 4. Check data types
    assert pd.api.types.is_integer_dtype(df["num_people"])
    assert pd.api.types.is_integer_dtype(df["photographer_experience_years"])
    assert pd.api.types.is_integer_dtype(df["lead_time_days"])
    assert pd.api.types.is_integer_dtype(df["actual_duration_minutes"])
    assert pd.api.types.is_float_dtype(df["quoted_price"])
    assert pd.api.types.is_float_dtype(df["true_wtp"])
    assert pd.api.types.is_bool_dtype(df["booking_accepted"])


def test_value_ranges():
    """
    Validates range bounds of categorical and numerical features.
    """
    df = generate_dataset(num_records=200, seed=42)
    
    # Shoot types
    valid_shoot_types = {"portrait", "event", "wedding", "graduation", "nature"}
    assert set(df["shoot_type"].unique()).issubset(valid_shoot_types)
    
    # Experience years: [1, 20]
    assert df["photographer_experience_years"].min() >= 1
    assert df["photographer_experience_years"].max() <= 20
    
    # Lead time: [1, 180]
    assert df["lead_time_days"].min() >= 1
    assert df["lead_time_days"].max() <= 180
    
    # Prices should be positive
    assert (df["quoted_price"] > 0).all()
    assert (df["true_wtp"] > 0).all()
    
    # Day of week
    valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    assert set(df["day_of_week"].unique()).issubset(valid_days)
    
    # Seasons
    valid_seasons = {"wedding_season", "graduation_season", "regular_season"}
    assert set(df["season"].unique()).issubset(valid_seasons)


def test_plausible_duration_bounds():
    """
    Checks that duration limits follow realistic parameters.
    """
    # Check portrait duration bounds
    for num in [1, 5]:
        for loc in ["indoor", "outdoor"]:
            dur = calculate_duration("portrait", num, loc)
            assert 15 <= dur <= 120, f"Portrait duration {dur} out of bounds"
            
    # Check wedding duration bounds
    for num in [30, 200]:
        for loc in ["indoor", "outdoor"]:
            dur = calculate_duration("wedding", num, loc)
            assert 180 <= dur <= 1000, f"Wedding duration {dur} out of bounds"


def test_pricing_compounding_effects():
    """
    Ensures compounding multipliers affect prices correctly in the expected directions.
    """
    # 1. Experience effect: 20 years vs 1 year (all else equal)
    p_high_exp, _, _ = calculate_price_and_acceptance(
        shoot_type="portrait", season="regular_season", day_of_week="Monday",
        lead_time_days=30, experience_years=20, num_people=1
    )
    p_low_exp, _, _ = calculate_price_and_acceptance(
        shoot_type="portrait", season="regular_season", day_of_week="Monday",
        lead_time_days=30, experience_years=1, num_people=1
    )
    assert p_high_exp > p_low_exp, "Experience premium not applied correctly"
    
    # 2. Season effect: wedding season vs regular season for wedding shoot
    p_peak, _, _ = calculate_price_and_acceptance(
        shoot_type="wedding", season="wedding_season", day_of_week="Monday",
        lead_time_days=30, experience_years=5, num_people=100
    )
    p_regular, _, _ = calculate_price_and_acceptance(
        shoot_type="wedding", season="regular_season", day_of_week="Monday",
        lead_time_days=30, experience_years=5, num_people=100
    )
    assert p_peak > p_regular, "Season premium not applied correctly"


def test_elasticity_acceptance_curve():
    """
    Verifies that client booking acceptance behaves elastically as price exceeds WTP.
    """
    df = generate_dataset(num_records=1000, seed=42)
    df["price_wtp_ratio"] = df["quoted_price"] / df["true_wtp"]
    
    low_ratio_bookings = df[df["price_wtp_ratio"] <= 0.85]
    high_ratio_bookings = df[df["price_wtp_ratio"] >= 1.15]
    
    low_accept_rate = low_ratio_bookings["booking_accepted"].mean()
    high_accept_rate = high_ratio_bookings["booking_accepted"].mean()
    
    print(f"Low ratio acceptance: {low_accept_rate:.2f}")
    print(f"High ratio acceptance: {high_accept_rate:.2f}")
    
    assert low_accept_rate > high_accept_rate, (
        f"Acceptance rate should be higher for lower price ratios: "
        f"low_ratio_rate={low_accept_rate:.2f}, high_ratio_rate={high_accept_rate:.2f}"
    )
    assert low_accept_rate > 0.60, f"Low ratio acceptance should be high, got {low_accept_rate:.2f}"
    assert high_accept_rate < 0.40, f"High ratio acceptance should be low, got {high_accept_rate:.2f}"
