"""
ShutterIQ - Dynamic Pricing & Elasticity Model

This script trains two models to implement a dynamic pricing engine:
1. Baseline Model: A Linear Regression pipeline that predicts the log of quoted price based on booking features.
2. Elasticity Model: A Logistic Regression pipeline that predicts booking acceptance P(Accept | price, features).

It then uses these models to:
- Maximize expected revenue: Price * P(Accept | price, features)
- Recover the implied willingness-to-pay (WTP) at the P(Accept) = 50% threshold and evaluate it against ground-truth.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score

# Define directory and file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bookings.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pricing_model.pkl")

# List of shoot types in the simulator
SHOOT_TYPES = ["portrait", "event", "wedding", "graduation", "nature"]


def load_data() -> pd.DataFrame:
    """
    Loads the synthetic bookings dataset. If it does not exist, runs the booking generator.
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


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates interaction terms between quoted_price and each shoot type.
    This allows the logistic regression model to learn a separate price elasticity
    coefficient for each photography service type, resolving price-scale mismatches.
    """
    df = df.copy()
    for stype in SHOOT_TYPES:
        df[f"price_x_{stype}"] = df["quoted_price"] * (df["shoot_type"] == stype)
    return df


def build_elasticity_pipeline() -> Pipeline:
    """
    Builds a pipeline for the Logistic Regression elasticity classifier.
    Note: We exclude 'day_of_week' and 'lead_time_days' from features here because they are
    exogenous price shifters (acting as instrumental variables). Including them introduces severe
    multicollinearity that renders the price coefficient unstable or positive.
    """
    categorical_features = ["shoot_type", "season"]
    interaction_cols = [f"price_x_{stype}" for stype in SHOOT_TYPES]
    numerical_features = ["photographer_experience_years", "num_people", "quoted_price"] + interaction_cols
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ]
    )
    
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(C=10.0, max_iter=1000, random_state=42))
    ])


def build_baseline_pipeline() -> Pipeline:
    """
    Builds a pipeline for the baseline price regressor.
    Predicts the historical quoted price from all booking features.
    """
    categorical_features = ["shoot_type", "season", "day_of_week"]
    numerical_features = ["lead_time_days", "photographer_experience_years", "num_people"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )
    
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])


def calculate_implied_wtp(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Calculates the implied willingness-to-pay (WTP) for each booking in X.
    The implied WTP is the price at which the booking acceptance probability is exactly 0.5.
    
    Under the interaction model:
    logit(P(Accept)) = score_other + w_price_total * price = 0
    where:
      w_price_total = w_price_base + w_price_interaction (for the booking's shoot type)
      score_other = intercept + sum(w_cat * X_cat) + sum(w_num * X_num) [excluding price terms]
    
    Solving for price gives: implied_wtp = - score_other / w_price_total
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    # Extract coefficients and intercept from trained model
    coef = classifier.coef_[0]
    intercept = classifier.intercept_[0]
    
    feature_names = list(preprocessor.get_feature_names_out())
    scaler = preprocessor.named_transformers_["num"]
    scaler_means = scaler.mean_
    scaler_scales = scaler.scale_
    
    interaction_cols = [f"price_x_{stype}" for stype in SHOOT_TYPES]
    num_features = ["photographer_experience_years", "num_people", "quoted_price"] + interaction_cols
    num_idx_in_scaler = {name: idx for idx, name in enumerate(num_features)}
    
    # Calculate unscaled coefficients and unscaled intercept
    unscaled_coef = np.zeros_like(coef)
    unscaled_intercept = intercept
    
    for idx, feat_name in enumerate(feature_names):
        if feat_name.startswith("cat__"):
            # Categorical features are not scaled
            unscaled_coef[idx] = coef[idx]
        elif feat_name.startswith("num__"):
            # Numerical features were standard-scaled
            raw_feat_name = feat_name.split("__")[1]
            scaler_idx = num_idx_in_scaler[raw_feat_name]
            
            mean_val = scaler_means[scaler_idx]
            scale_val = scaler_scales[scaler_idx]
            
            unscaled_coef[idx] = coef[idx] / scale_val
            unscaled_intercept -= coef[idx] * mean_val / scale_val
            
    # Transform input DataFrame and unscale numerical features to get unscaled design matrix
    X_trans = preprocessor.transform(X)
    X_trans_unscaled = X_trans.copy()
    for idx, feat_name in enumerate(feature_names):
        if feat_name.startswith("num__"):
            raw_feat_name = feat_name.split("__")[1]
            scaler_idx = num_idx_in_scaler[raw_feat_name]
            mean_val = scaler_means[scaler_idx]
            scale_val = scaler_scales[scaler_idx]
            X_trans_unscaled[:, idx] = X_trans[:, idx] * scale_val + mean_val
            
    # Find price term indices in preprocessor feature names
    price_base_idx = feature_names.index("num__quoted_price")
    w_price_base = unscaled_coef[price_base_idx]
    
    # Map shoot types to their interaction term feature index
    interaction_indices = {}
    for stype in SHOOT_TYPES:
        col_name = f"num__price_x_{stype}"
        if col_name in feature_names:
            interaction_indices[stype] = feature_names.index(col_name)
            
    # Identify indices of non-price features (to calculate score_other)
    non_price_indices = []
    for idx, feat_name in enumerate(feature_names):
        is_price_term = (idx == price_base_idx) or any(idx == interaction_indices[stype] for stype in interaction_indices)
        if not is_price_term:
            non_price_indices.append(idx)
            
    # Calculate score_other matrix multiplication
    X_non_price = X_trans_unscaled[:, non_price_indices]
    coef_non_price = unscaled_coef[non_price_indices]
    score_others = unscaled_intercept + X_non_price @ coef_non_price
    
    # Compute implied WTP for each row
    implied_wtp = np.zeros(len(X))
    X_reset = X.reset_index(drop=True)
    for i, row in X_reset.iterrows():
        stype = row["shoot_type"]
        w_price_total = w_price_base
        if stype in interaction_indices:
            w_price_total += unscaled_coef[interaction_indices[stype]]
            
        if abs(w_price_total) < 1e-6:
            raise ValueError(f"Price coefficient for shoot type {stype} is near-zero.")
            
        implied_wtp[i] = - score_others[i] / w_price_total
        
    return implied_wtp


def optimize_price(
    elasticity_pipeline: Pipeline,
    baseline_pipeline: Pipeline,
    booking_features: dict,
    price_grid_points: int = 151
) -> float:
    """
    Finds the recommended price that maximizes expected revenue for a single booking request.
    Expected Revenue(p) = p * P(Accept | p, features)
    """
    input_df = pd.DataFrame([booking_features])
    
    # Extract features required by baseline model
    baseline_features = ["shoot_type", "season", "day_of_week", "lead_time_days", "photographer_experience_years", "num_people"]
    baseline_input = input_df[baseline_features]
    
    # Predict baseline price (exponetial of predicted log price)
    pred_baseline_log = float(baseline_pipeline.predict(baseline_input)[0])
    pred_baseline = np.exp(pred_baseline_log)
    
    # Define search range centered around the predicted baseline price
    low_bound = max(10.0, 0.5 * pred_baseline)
    high_bound = 2.0 * pred_baseline
    candidate_prices = np.linspace(low_bound, high_bound, price_grid_points)
    
    # Commercial round to nearest $5
    candidate_prices = np.round(candidate_prices / 5.0) * 5.0
    candidate_prices = np.unique(candidate_prices)
    
    # Create prediction dataframe for all candidate prices
    candidate_df = pd.DataFrame([booking_features] * len(candidate_prices))
    candidate_df["quoted_price"] = candidate_prices
    
    # Add interaction features before predicting
    candidate_df = add_interaction_features(candidate_df)
    
    # Predict acceptance probabilities
    probs = elasticity_pipeline.predict_proba(candidate_df)[:, 1]
    
    # Find price that maximizes expected revenue
    expected_revenues = candidate_prices * probs
    best_idx = np.argmax(expected_revenues)
    
    return float(candidate_prices[best_idx])


def optimize_prices_batch(
    elasticity_pipeline: Pipeline,
    baseline_pipeline: Pipeline,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Helper function to optimize prices for a batch of bookings.
    """
    recommended_prices = []
    for _, row in X.iterrows():
        rec_p = optimize_price(elasticity_pipeline, baseline_pipeline, row.to_dict())
        recommended_prices.append(rec_p)
    return np.array(recommended_prices)


def train_models():
    """
    Loads data, trains baseline regression and elasticity classification models,
    evaluates them (including WTP recovery), and saves the model artifacts.
    """
    print("Loading bookings dataset...")
    df = load_data()
    
    # Define features and targets
    features = ["shoot_type", "season", "day_of_week", "lead_time_days", "photographer_experience_years", "num_people"]
    
    # Add interaction columns to the raw data
    df_interacted = add_interaction_features(df)
    
    # Splitting data (using identical splits for consistency)
    # Split using df_interacted
    X_train_raw, X_test_raw, y_class_train, y_class_test = train_test_split(
        df_interacted[features + ["quoted_price"] + [f"price_x_{s}" for s in SHOOT_TYPES]],
        df_interacted["booking_accepted"],
        test_size=0.2,
        random_state=42
    )
    
    # Baseline regressor features exclude quoted_price and interaction terms
    X_reg_train = X_train_raw[features]
    X_reg_test = X_test_raw[features]
    
    # Train on natural logarithm of price
    y_reg_train = np.log(df_interacted.loc[X_train_raw.index, "quoted_price"])
    y_reg_test_original = df_interacted.loc[X_test_raw.index, "quoted_price"]
    
    print("\n--- Training Baseline Model (Log-Price Regression) ---")
    baseline_pipeline = build_baseline_pipeline()
    baseline_pipeline.fit(X_reg_train, y_reg_train)
    
    # Evaluate Baseline Model in original space
    baseline_pred_log = baseline_pipeline.predict(X_reg_test)
    baseline_pred = np.exp(baseline_pred_log)
    baseline_mae = mean_absolute_error(y_reg_test_original, baseline_pred)
    baseline_r2 = r2_score(y_reg_test_original, baseline_pred)
    print(f"Baseline Regressor Test MAE: ${baseline_mae:.2f}")
    print(f"Baseline Regressor Test R² : {baseline_r2:.4f}")
    
    print("\n--- Training Elasticity Model (Logistic Regression with Interactions) ---")
    # For the elasticity classifier, we drop day_of_week and lead_time_days from X
    classifier_cols = ["shoot_type", "season", "photographer_experience_years", "num_people", "quoted_price"] + [f"price_x_{s}" for s in SHOOT_TYPES]
    X_class_train = X_train_raw[classifier_cols]
    X_class_test = X_test_raw[classifier_cols]
    
    elasticity_pipeline = build_elasticity_pipeline()
    elasticity_pipeline.fit(X_class_train, y_class_train)
    
    # Evaluate Elasticity Classifier
    class_pred = elasticity_pipeline.predict(X_class_test)
    class_probs = elasticity_pipeline.predict_proba(X_class_test)[:, 1]
    
    accuracy = accuracy_score(y_class_test, class_pred)
    auc = roc_auc_score(y_class_test, class_probs)
    print(f"Elasticity Classifier Test Accuracy: {accuracy:.4f}")
    print(f"Elasticity Classifier Test ROC AUC : {auc:.4f}")
    
    # WTP Recovery Evaluation
    print("\n--- Willingness-to-Pay (WTP) Recovery Evaluation ---")
    true_wtp_test = df_interacted.loc[X_test_raw.index, "true_wtp"].values
    
    # Calculate implied WTP from model coefficients
    implied_wtp_test = calculate_implied_wtp(elasticity_pipeline, X_class_test)
    
    overall_wtp_mae = mean_absolute_error(true_wtp_test, implied_wtp_test)
    overall_wtp_r2 = r2_score(true_wtp_test, implied_wtp_test)
    overall_wtp_corr = np.corrcoef(true_wtp_test, implied_wtp_test)[0, 1]
    
    print(f"Overall WTP Recovery MAE: ${overall_wtp_mae:.2f}")
    print(f"Overall WTP Recovery R² : {overall_wtp_r2:.4f}")
    print(f"Overall WTP Correlation : {overall_wtp_corr:.4f}")
    
    print("\nDetailed WTP Recovery by Shoot Type:")
    X_test_with_wtp = X_test_raw.copy()
    X_test_with_wtp["true_wtp"] = true_wtp_test
    X_test_with_wtp["implied_wtp"] = implied_wtp_test
    
    wtp_by_type = {}
    for stype in SHOOT_TYPES:
        sub = X_test_with_wtp[X_test_with_wtp["shoot_type"] == stype]
        if len(sub) > 0:
            sub_mae = mean_absolute_error(sub["true_wtp"], sub["implied_wtp"])
            sub_r2 = r2_score(sub["true_wtp"], sub["implied_wtp"])
            sub_corr = np.corrcoef(sub["true_wtp"], sub["implied_wtp"])[0, 1]
            print(f"  - {stype:12s}: MAE = ${sub_mae:7.2f} | R² = {sub_r2:7.4f} | Corr = {sub_corr:7.4f}")
            wtp_by_type[stype] = {"mae": sub_mae, "r2": sub_r2, "corr": sub_corr}
        else:
            wtp_by_type[stype] = {"mae": np.nan, "r2": np.nan, "corr": np.nan}
            
    # Save artifacts
    print(f"\nSaving trained models and metrics to {MODEL_PATH}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = {
        "elasticity_pipeline": elasticity_pipeline,
        "baseline_pipeline": baseline_pipeline,
        "metrics": {
            "baseline": {"mae": baseline_mae, "r2": baseline_r2},
            "elasticity": {"accuracy": accuracy, "auc": auc},
            "wtp_recovery": {
                "overall_mae": overall_wtp_mae,
                "overall_r2": overall_wtp_r2,
                "overall_corr": overall_wtp_corr,
                "by_type": wtp_by_type
            }
        }
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifacts, f)
        
    print("Pricing model training complete.")


# Global cache for loaded model to speed up predictions
_LOADED_ARTICIFS = None

def _get_artifacts():
    global _LOADED_ARTICIFS
    if _LOADED_ARTICIFS is None:
        if not os.path.exists(MODEL_PATH):
            train_models()
        with open(MODEL_PATH, "rb") as f:
            _LOADED_ARTICIFS = pickle.load(f)
    return _LOADED_ARTICIFS


def recommend_price_and_expected_revenue(booking_features: dict) -> tuple[float, float]:
    """
    Given booking features, recommends the price maximizing expected revenue,
    and returns both the recommended price and the corresponding expected revenue.
    """
    artifacts = _get_artifacts()
    elasticity_pipeline = artifacts["elasticity_pipeline"]
    baseline_pipeline = artifacts["baseline_pipeline"]
    
    # Recommended Price
    rec_price = optimize_price(elasticity_pipeline, baseline_pipeline, booking_features)
    
    # Calculate Expected Revenue
    candidate_df = pd.DataFrame([booking_features])
    candidate_df["quoted_price"] = rec_price
    candidate_df = add_interaction_features(candidate_df)
    
    # We need to filter features to what the classifier expects
    classifier_cols = ["shoot_type", "season", "photographer_experience_years", "num_people", "quoted_price"] + [f"price_x_{s}" for s in SHOOT_TYPES]
    prob_accept = float(elasticity_pipeline.predict_proba(candidate_df[classifier_cols])[0, 1])
    expected_revenue = rec_price * prob_accept
    
    return rec_price, expected_revenue


if __name__ == "__main__":
    train_models()
