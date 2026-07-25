# ShutterIQ 📸🤖

An intelligent scheduling assistant and business optimization suite for photographers. ShutterIQ uses machine learning to predict shoot durations, recommend dynamic pricing based on client willingness-to-pay (WTP), and optimize outdoor shoot windows using local weather and solar golden hour alignment.

This repository is designed as a data science portfolio project demonstrating the integration of predictive models, pricing optimization, and smart scheduling heuristics.

---

## Project Architecture & Core Components

ShutterIQ is structured around three primary data-driven engines:

### 1. Predictive Shoot Duration Engine
* **Objective:** Predict actual shoot duration (regression) to optimize photographer timetables and prevent overlapping bookings.
* **Approach:** Uses factors like shoot type (portrait, wedding, event, graduation, nature), client group size (non-linear logarithmic scaling), location type (indoor vs. outdoor), and weather conditions to forecast required calendar slots, incorporating historical variance.

### 2. Dynamic Pricing Engine
* **Objective:** Recommend optimized, yield-maximizing prices for photography booking requests.
* **Approach:** Estimates price elasticity of demand by comparing quoted price against the client's latent willingness-to-pay (WTP) threshold. Combines compounding multipliers for seasonality (peaks in Sri Lankan wedding and graduation months), day-of-week demand spikes, lead time constraints (last-minute booking markup), and photographer experience.

### 3. Golden Hour & Weather Optimizer
* **Objective:** Recommend scheduling windows for outdoor shoots to maximize lighting quality.
* **Approach:** Integrates weather forecasts and solar elevation calculations (golden hour peaks) to determine the best hours of the day for outdoor portraiture, graduation, and nature shoots.

---

## Project Structure

```text
shutteriq/
  ├── data/         # Generated synthetic booking datasets (ignored in git)
  ├── simulate/     # Data simulation engine
  │     ├── __init__.py
  │     └── generate_bookings.py  # Synthetic booking generator with economic modeling
  ├── models/       # ML models (predictors and elasticity estimators) - Placeholder
  ├── api/          # FastAPI backend services - Placeholder
  ├── app/          # Streamlit user interface - Placeholder
  ├── tests/        # Pytest unit tests
  │     ├── __init__.py
  │     └── test_generate_bookings.py
  ├── .gitignore
  ├── requirements.txt
  └── CHANGELOG.md
```

---

## Stage 0 Scope: Scaffolding & Synthetic Data Foundation
In Stage 0, we establish the project structure, development environment, and the synthetic data generator. The generator simulates realistic booking transactions under the following economic and physical assumptions:
* ** Sri Lankan Seasonality:**
  * **Wedding Season Peak:** May, June, November, and December (applying a $+35\%$ multiplier to pricing and willingness-to-pay).
  * **Graduation Season Peak:** July and August (applying a $+25\%$ multiplier to graduation shoots).
* **Non-linear Durations:** Duration scales logarithmically with group size ($\alpha \log(N)$) and adds $+15\%$ setup overhead for outdoor shoots, with multiplicative log-normal noise.
* **Price Elasticity:** Customer acceptance of quotes is modeled using a logistic curve representing price-to-willingness-to-pay ratios ($P(\text{Accept}) = \frac{1}{1 + e^{k(\text{price}/\text{wtp} - 1)}}$). This ensures that higher quotes relative to a customer's budget result in fewer accepted bookings, creating a realistic demand curve.

---

## Stage 1: Predictive Shoot Duration Model
In Stage 1, we implemented the **Predictive Shoot Duration Engine** using machine learning to predict actual shoot durations from booking parameters.

### 1. Modeling & Validation Results
We split our 1,000 synthetic records into an 80/20 train/test split and evaluated both a baseline Linear Regression model and a challenger XGBoost Regressor:

| Model | Train MAE | Train RMSE | Test MAE | Test RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | 23.12 mins | 35.63 mins | 20.31 mins | 29.94 mins |
| **XGBoost Regressor (Challenger)** | 11.83 mins | 19.52 mins | 16.16 mins | 31.31 mins |

*Observations:* XGBoost improves mean absolute error (MAE) significantly from 20.31 minutes to 16.16 minutes on the test set, capturing non-linear relationships (logarithmic group size scaling and interaction terms) that Linear Regression struggles to approximate.

### 2. Recommended Scheduling Buffer Time
Rather than utilizing an arbitrary flat buffer (e.g. "always add 30 mins"), we estimate a heteroscedastic scheduling buffer based on the prediction uncertainty (standard deviation of residuals, $e = y - \hat{y}$) for each shoot type. Using a $1.645\times\sigma_{e, \text{type}}$ multiplier ensures a **95% protection threshold** against scheduling overruns:
* **Wedding:** 140 minutes buffer (reflects high schedule volatility and guest counts)
* **Event:** 50 minutes buffer
* **Nature:** 25 minutes buffer
* **Graduation:** 15 minutes buffer
* **Portrait:** 10 minutes buffer

### 3. Model Interpretability (SHAP Analysis)
We computed SHAP values on the XGBoost model to inspect feature impacts:
![SHAP Summary Plot](shutteriq/reports/duration_model_shap.png)

*Key Insights:*
* **Guest Count (`num_people`):** Highly positive impact. Larger groups represent the single largest driver of extended shoot durations.
* **Shoot Type:** Wedding and event shoot types have strong positive base effects on duration, while portrait and graduation have negative effects relative to the reference category.
* **Location Type (`location_type_outdoor`):** Outdoor shoots shift predictions upward due to ambient lighting adjustments and setup overhead.

---

## Installation & Setup

Ensure you have Python 3.11+ installed.

### 1. Set Up the Virtual Environment
Create and activate a local Python virtual environment:
```bash
# Create venv
python -m venv .venv

# Activate venv (macOS/Linux)
source .venv/bin/activate

# Activate venv (Windows)
.venv\Scripts\activate
```

### 2. Install Dependencies
Install all package requirements pinned to ensure reproducibility:
```bash
pip install -r requirements.txt
```

### 3. Generate the Synthetic Dataset
Run the data generator to create a baseline set of 1,000 bookings:
```bash
python -m shutteriq.simulate.generate_bookings --num-records 1000 --output shutteriq/data/bookings.csv
```

### 4. Run the Test Suite
Validate the synthetic data generation engine and sanity checks:
```bash
pytest
```
