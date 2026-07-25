"""
ShutterIQ - Model Explainability

This script computes SHAP (SHapley Additive exPlanations) values for the trained
XGBoost shoot duration model and saves a summary plot for documentation.
"""

import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Define directory and file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "shutteriq", "data", "bookings.csv")
MODEL_PATH = os.path.join(BASE_DIR, "shutteriq", "models", "duration_model.pkl")
REPORT_DIR = os.path.join(BASE_DIR, "shutteriq", "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "duration_model_shap.png")


def run_explainability():
    """
    Loads the trained model pipeline and data, performs preprocessing transformation,
    calculates SHAP values, and saves a SHAP summary beeswarm plot.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please run duration_model.py first.")
        
    print("Loading model artifacts...")
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)
        
    xgb_pipeline = artifacts["xgb_pipeline"]
    features = artifacts["features"]
    
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    X = df[features]
    
    # Extract preprocessor and model from pipeline
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    xgb_model = xgb_pipeline.named_steps["regressor"]
    
    print("Pre-processing features for SHAP explainability...")
    X_encoded = preprocessor.transform(X)
    
    # Retrieve clean feature names out of the ColumnTransformer
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.replace("cat__", "").replace("num__", "") for name in feature_names]
    
    X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)
    
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_encoded_df)
    
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    
    # Generate beeswarm summary plot
    shap.summary_plot(shap_values, X_encoded_df, show=False)
    
    plt.title("SHAP Feature Impact on Predicted Shoot Duration (Minutes)", fontsize=14, pad=15)
    plt.tight_layout()
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    plt.savefig(REPORT_PATH, dpi=150)
    plt.close()
    
    print(f"SHAP summary plot successfully saved to {REPORT_PATH}")


if __name__ == "__main__":
    run_explainability()
