"""
ShutterIQ - Shoot Duration Prediction Model

This script trains a predictive model to forecast actual shoot durations (in minutes)
using booking attributes. It compares a baseline Linear Regression model with an XGBoost Regressor.
It also derives a category-specific recommended buffer time based on prediction uncertainty (residuals).
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb

# Define directory and file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "bookings.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "duration_model.pkl")


def load_data() -> pd.DataFrame:
    """
    Loads the synthetic dataset. If bookings.csv does not exist, runs the simulator
    to generate a default dataset first.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Generating default synthetic bookings...")
        from simulate.generate_bookings import generate_dataset
        df = generate_dataset(num_records=1000, seed=42)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


def build_pipeline(regressor) -> Pipeline:
    """
    Builds a scikit-learn Pipeline with preprocessing and the target regressor.
    """
    categorical_features = ["shoot_type", "location_type", "lighting_setup"]
    numerical_features = ["num_people", "photographer_experience_years"]
    
    # We drop the first category level to avoid multicollinearity for linear models,
    # and handle unknown categories during inference gracefully.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )
    
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])


def calculate_buffers_by_type(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    confidence_multiplier: float = 1.645
) -> tuple[dict[str, int], int]:
    """
    Calculates recommended buffer minutes per shoot type from prediction uncertainty (residuals).
    
    Why this approach (Option A)?
    - Shoots are subject to different operational risks. Wedding durations vary wildly (e.g., standard dev
      of prediction errors might be 40 minutes) whereas portraits are highly consistent (std dev ~ 5 minutes).
    - Under a normality assumption, a multiplier of 1.645 captures the one-sided 95% upper limit of errors,
      ensuring the actual shoot will be completed within (Predicted + Buffer) 95% of the time.
    - We round buffers UP to the nearest 5 minutes for scheduling practicality.
    """
    test_df = test_df.copy()
    test_df["residual"] = y_test - y_pred
    
    # Group residuals by shoot_type and calculate standard deviation
    std_by_type = test_df.groupby("shoot_type")["residual"].std().to_dict()
    
    buffers = {}
    for stype, std_err in std_by_type.items():
        if pd.isna(std_err):
            std_err = 10.0  # Safe default if partition is too small
        
        # Calculate raw buffer minutes (1.645 standard deviations)
        raw_buffer = confidence_multiplier * std_err
        
        # Round up to nearest 5 minutes for commercial booking convenience
        rounded_buffer = int(np.ceil(max(5.0, raw_buffer) / 5.0) * 5.0)
        buffers[stype] = rounded_buffer
        
    # Calculate overall baseline buffer for fallbacks
    overall_std = test_df["residual"].std()
    overall_buffer = int(np.ceil((confidence_multiplier * overall_std) / 5.0) * 5.0)
    
    return buffers, overall_buffer


def train_models():
    """
    Loads data, splits into train/test, trains baseline (Linear Regression)
    and challenger (XGBoost) models, prints metrics, and saves model artifacts.
    """
    print("Loading dataset...")
    df = load_data()
    
    # Filter features and target
    features = ["shoot_type", "num_people", "location_type", "lighting_setup", "photographer_experience_years"]
    target = "actual_duration_minutes"
    
    X = df[features]
    y = df[target]
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Training Baseline (Linear Regression) ---")
    lr_pipeline = build_pipeline(LinearRegression())
    lr_pipeline.fit(X_train, y_train)
    
    # Evaluate Baseline
    lr_train_pred = lr_pipeline.predict(X_train)
    lr_test_pred = lr_pipeline.predict(X_test)
    
    lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
    lr_train_rmse = root_mean_squared_error(y_train, lr_train_pred)
    lr_test_mae = mean_absolute_error(y_test, lr_test_pred)
    lr_test_rmse = root_mean_squared_error(y_test, lr_test_pred)
    
    print(f"Linear Regression Train - MAE: {lr_train_mae:.2f} mins, RMSE: {lr_train_rmse:.2f} mins")
    print(f"Linear Regression Test  - MAE: {lr_test_mae:.2f} mins, RMSE: {lr_test_rmse:.2f} mins")
    
    print("\n--- Training Challenger (XGBoost Regressor) ---")
    # Using sensible default hyper-parameters for tabular data
    xgb_reg = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        random_state=42
    )
    xgb_pipeline = build_pipeline(xgb_reg)
    xgb_pipeline.fit(X_train, y_train)
    
    # Evaluate Challenger
    xgb_train_pred = xgb_pipeline.predict(X_train)
    xgb_test_pred = xgb_pipeline.predict(X_test)
    
    xgb_train_mae = mean_absolute_error(y_train, xgb_train_pred)
    xgb_train_rmse = root_mean_squared_error(y_train, xgb_train_pred)
    xgb_test_mae = mean_absolute_error(y_test, xgb_test_pred)
    xgb_test_rmse = root_mean_squared_error(y_test, xgb_test_pred)
    
    print(f"XGBoost Regressor Train - MAE: {xgb_train_mae:.2f} mins, RMSE: {xgb_train_rmse:.2f} mins")
    print(f"XGBoost Regressor Test  - MAE: {xgb_test_mae:.2f} mins, RMSE: {xgb_test_rmse:.2f} mins")
    
    # Calculate recommended scheduling buffers
    print("\nDeriving scheduling buffers from prediction residuals...")
    buffers, overall_buffer = calculate_buffers_by_type(X_test, y_test, xgb_test_pred)
    print("Recommended buffers (95% protection threshold):")
    for stype, buf in buffers.items():
        print(f"  - {stype:12s}: {buf:3d} mins buffer")
    print(f"  - Fallback Overall: {overall_buffer:3d} mins buffer")
    
    # Save artifacts
    print(f"\nSaving trained model and assets to {MODEL_PATH}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = {
        "xgb_pipeline": xgb_pipeline,
        "lr_pipeline": lr_pipeline,
        "buffers": buffers,
        "overall_buffer": overall_buffer,
        "features": features,
        "metrics": {
            "lr": {"test_mae": lr_test_mae, "test_rmse": lr_test_rmse},
            "xgb": {"test_mae": xgb_test_mae, "test_rmse": xgb_test_rmse}
        }
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifacts, f)
        
    print("Model training complete.")


# Global cache for loaded model to speed up predictions
_LOADED_ARTICIFS = None

def _get_artifacts():
    global _LOADED_ARTICIFS
    if _LOADED_ARTICIFS is None:
        if not os.path.exists(MODEL_PATH):
            # Proactively train models if they haven't been trained yet
            train_models()
        with open(MODEL_PATH, "rb") as f:
            _LOADED_ARTICIFS = pickle.load(f)
    return _LOADED_ARTICIFS


def predict_duration_with_buffer(
    shoot_type: str,
    num_people: int,
    location_type: str,
    lighting_setup: str,
    photographer_experience_years: int
) -> tuple[float, int]:
    """
    Predicts the expected shoot duration and returns a recommended scheduling buffer.
    
    Returns:
        tuple[predicted_duration_minutes, recommended_buffer_minutes]
    """
    artifacts = _get_artifacts()
    xgb_pipeline = artifacts["xgb_pipeline"]
    buffers = artifacts["buffers"]
    overall_buffer = artifacts["overall_buffer"]
    
    # Build single row dataframe matching feature order
    input_df = pd.DataFrame([{
        "shoot_type": shoot_type,
        "num_people": num_people,
        "location_type": location_type,
        "lighting_setup": lighting_setup,
        "photographer_experience_years": photographer_experience_years
    }])
    
    pred_val = float(xgb_pipeline.predict(input_df)[0])
    
    # Ensure positive prediction limit
    pred_val = max(15.0, pred_val)
    
    # Retrieve buffer based on shoot type
    buffer_val = buffers.get(shoot_type, overall_buffer)
    
    return pred_val, buffer_val


if __name__ == "__main__":
    train_models()
