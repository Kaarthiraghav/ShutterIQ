"""
ShutterIQ App - Formatting and Analytical Helper Utilities

This module contains extracted, pure formatting and SHAP-based analytical helpers.
These are decoupled from the Streamlit UI script to ensure they are fully unit-testable.
"""

import os
import pickle
import numpy as np
import pandas as pd


def format_duration(minutes: float) -> str:
    """
    Converts a duration in minutes into a clean, human-readable string.
    
    Examples:
        45.0  -> "45 mins"
        60.0  -> "1 hr"
        75.0  -> "1 hr 15 mins"
        120.0 -> "2 hrs"
        150.0 -> "2 hrs 30 mins"
    """
    minutes_int = int(round(minutes))
    if minutes_int < 60:
        return f"{minutes_int} mins"
    
    hours = minutes_int // 60
    rem_mins = minutes_int % 60
    
    h_str = "1 hr" if hours == 1 else f"{hours} hrs"
    
    if rem_mins == 0:
        return h_str
    else:
        return f"{h_str} {rem_mins} mins"


def get_weather_badge(score: float) -> str:
    """
    Returns an emoji badge representing the quality classification of a slot score.
    
    Ranges:
        score >= 0.8:       "🌟 Excellent"
        0.6 <= score < 0.8: "✨ Good"
        0.4 <= score < 0.6: "⛅ Fair"
        score < 0.4:        "🌧️ Poor"
    """
    if score >= 0.8:
        return "🌟 Excellent"
    elif score >= 0.6:
        return "✨ Good"
    elif score >= 0.4:
        return "⛅ Fair"
    else:
        return "🌧️ Poor"


def explain_duration_factors(
    shoot_type: str,
    num_people: int,
    location_type: str,
    lighting_setup: str,
    photographer_experience_years: int
) -> dict:
    """
    Computes tree-based SHAP attributions for the single prediction request.
    Groups the 10 preprocessed columns back into 5 original input variables.
    
    Returns:
        A dict containing:
          - base_value: The average predicted duration across the training set (~249 mins).
          - prediction: The predicted duration.
          - factors: A dict mapping human-readable category offsets to float values (minutes).
    """
    import shap
    from models.duration_model import MODEL_PATH

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained duration model artifact not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)

    xgb_pipeline = artifacts["xgb_pipeline"]
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    regressor = xgb_pipeline.named_steps["regressor"]

    # Construct single-row input DataFrame
    input_df = pd.DataFrame([{
        "shoot_type": shoot_type,
        "num_people": num_people,
        "location_type": location_type,
        "lighting_setup": lighting_setup,
        "photographer_experience_years": photographer_experience_years
    }])

    # Preprocess
    X_trans = preprocessor.transform(input_df)

    # Initialize TreeExplainer and compute SHAP attributions
    explainer = shap.TreeExplainer(regressor)
    shap_vals = explainer.shap_values(X_trans)[0]

    # Map categorical features:
    # 0..3: cat__shoot_type_* (graduation, nature, portrait, wedding)
    # 4: cat__location_type_outdoor
    # 5..7: cat__lighting_setup_* (none, speedlight, studio_lights)
    # 8: num__num_people
    # 9: num__photographer_experience_years
    factors = {
        "Shoot Type (Service)": float(sum(shap_vals[0:4])),
        "Location Setting": float(shap_vals[4]),
        "Lighting Setup": float(sum(shap_vals[5:8])),
        "Group Size (People)": float(shap_vals[8]),
        "Photographer Experience": float(shap_vals[9])
    }

    return {
        "base_value": float(explainer.expected_value),
        "prediction": float(max(15.0, regressor.predict(X_trans)[0])),
        "factors": factors
    }
