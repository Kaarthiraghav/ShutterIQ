# ShutterIQ: Intelligent Scheduling & Pricing Assistant for Photographers

ShutterIQ is an intelligent booking, pricing, and scheduling assistant designed to help professional photographers optimize session durations, maximize dynamic revenue, and secure ideal natural lighting conditions. The system features a predictive shoot duration model (XGBoost) with category-specific operational safety buffers, an elasticity-based dynamic pricing engine (Logistic Regression) using causal inference controls to recover client willingness-to-pay (WTP), and a golden hour and weather scheduling optimizer (Astral + Open-Meteo API) with a rain-veto composite scorer. Exposing these models behind a unified FastAPI gateway and a polished, interactive Streamlit frontend dashboard, ShutterIQ converts raw booking inquiries into structured, optimized schedule recommendations.

---

## 1. Why This Project?

Traditional photography scheduling relies on static, flat-rate pricing lists and rigid "one-size-fits-all" duration slots (e.g., standardizing every portrait session to exactly 1 hour). This creates significant operational and commercial inefficiencies:
* **Over-allocation & Under-allocation:** A large group shoot outdoors requires additional buffer times for transit, setup, and posing, whereas a single person indoor shoot finishes ahead of schedule. Static scheduling results in double-bookings or wasted vacant slots.
* **Lost Pricing Yield:** Photographer demand is highly seasonal (weddings peak in summer/winter, graduations peak in spring/autumn) and fluctuates by day (weekends are highly valued). Fixed pricing fails to capture peak willingness-to-pay.
* **Selection Bias in Historical Pricing:** Training pricing models on historical sales data introduces a classic **endogeneity trap** (selection bias): photographers historically charge more during high-demand weekends and peak seasons, causing standard regression models to mistakenly learn that higher prices *drive* higher conversion rates (yielding a positive price coefficient).

ShutterIQ solves these problems by applying structured machine learning models, causal instruments, and geographical heuristics.

---

## 2. Architecture Overview

ShutterIQ is structured as a decoupled three-tier application:
1. **Frontend Dashboard (Streamlit):** Offers a polished user interface for entry of booking requests, dynamic API gateway status indicator, and rich Plotly visualizations of model explanations.
2. **REST API Layer (FastAPI):** Exposes four routes for predictions and a unified endpoint combining all three models, running model pre-caching on lifespan startup.
3. **Core ML & Heuristics Engines:** Executed locally in memory.

```text
                       +-------------------+
                       |   Streamlit App   |
                       | (Frontend/Client) |
                       +---------+---------+
                                 |
                                 v  (POST /schedule-suggestion)
                       +---------+---------+
                       |    FastAPI App    |
                       | (API Integration) |
                       +----+----+----+----+
                            |    |    |
       +--------------------+    |    +--------------------+
       |                         |                         |
       v                         v                         v
+------+------+           +------+------+           +------+------+
|  Duration   |           |   Dynamic   |           | Golden Hour |
|  Predictor  |           |   Pricing   |           |  & Weather  |
|  (XGBoost)  |           |  (Elastic)  |           |  Optimizer  |
+-------------+           +-------------+           +-------------+
```

---

## 3. Methodology per Component

### 3.1. Predictive Shoot Duration Model
* **Model Pipeline:** Built as an `XGBRegressor` pipeline preprocessed via a scikit-learn `ColumnTransformer` (one-hot encoding categorical variables like lighting setup and shoot type, passing through numerical variables).
* **Non-Linear Simulation Assumptions:** The model fits non-linear patterns, capturing that shoot duration scales logarithmically with group size ($1 + \alpha \log(N)$) and commands a $+15\%$ premium for outdoor logistics.
* **Category-Specific Operational Buffers:** Rather than an overall buffer, ShutterIQ computes the standard deviation of residuals (errors) *per shoot category* on test data. Using a $95\%$ one-sided confidence interval ($1.645 \times \sigma_{\text{category}}$), it rounds up buffers to the nearest 5 minutes. Highly variable shoots (e.g., weddings) get larger buffers, while predictable shoots (e.g., portraits) get shorter buffers.

### 3.2. Elasticity-Based Dynamic Pricing (Causal WTP Recovery)
The dynamic pricing engine recommends the revenue-maximizing price ($p^*$) that maximizes Expected Revenue $= p \times P(\text{Accept} \mid p)$.
* **Baseline Log-Space Pricing:** A linear regression is trained on $\log(\text{price})$ to predict the normal market price. Taking the exponent guarantees that recommended baseline rates are strictly positive.
* **Causal Inference Framing (Avoiding Selection Bias):** Day of the week, season, and lead time act as exogenous price shifters in the booking simulator, driving the photographer's quote rate but not the client's latent valuation. By **excluding** these shifters from the conversion classifier (treating them as instrumental variables), we block the endogeneity back-door path, successfully learning a negative price coefficient (downward-sloping demand curve).
* **Interaction Elasticity Classifier:** Fits a `LogisticRegression` classifier with interaction terms (`price * shoot_type`) and standard scaling. This allows the model to map separate price-sensitivity curves for each category, accommodating different budget scales ($150 portraits vs. $2,200 weddings).
* **WTP Recovery Evaluation Story (Synthetic Ground Truth):**
  We validate ShutterIQ's causal framework by evaluating the recovered willingness-to-pay (the price threshold at which $P(\text{Accept}) = 0.5$) against the simulator's true latent `WTP` in the test set.
  * **WTP Recovery R² Score:** **$0.9838$**
  * **WTP Recovery Correlation:** **$0.9920$**
  * This demonstrates that ShutterIQ's causal controls successfully recover the hidden demand curve.

### 3.3. Golden Hour & Weather Scheduling Optimizer
* **Solar Geometry (Astral):** Computes exact timezone-aware morning/evening golden hour and blue hour windows for any location (latitude/longitude) and date. Proximity to windows decays exponentially ($e^{-d/120}$ minutes).
* **Meteorological Forecasts (Open-Meteo):** Integrates live hourly forecasts (cloud cover %, rain probability %, temperature) from the Open-Meteo API.
* **Photographer Scoring Curves:**
  * *Cloud Cover:* Optimized for diffused light. Sweet spot is $30\%$ cover (score $1.0$); harsh direct sun ($0\%$ clouds) drops to $0.6$; overcast storm clouds ($100\%$ clouds) drops to $0.4$.
  * *Rain Veto:* Multiplicative factor $S_{\text{precip}} = 1 - (P_{\text{rain}}/100)$. A $100\%$ chance of rain vetoes the slot entirely, yielding a score of $0.0$.
* **Graceful Degradation:** If the Open-Meteo API times out or fails, the scheduler logs a warning and fallback-degrades to astronomical solar calculations, setting weather scores to $1.0$ (neutral).

---

## 4. Tech Stack

* **Machine Learning & Math:** `scikit-learn`, `xgboost`, `numpy`, `pandas`, `shap` (SHAP-based model explanations)
* **Astronomical Calculations:** `astral`
* **API Gateway:** `fastapi`, `uvicorn`, `pydantic` (validation schemas), `httpx` (client testing)
* **Frontend UI:** `streamlit`, `plotly` (interactive charts), `requests`
* **Testing:** `pytest`

---

## 5. Setup & Execution

### 5.1. Clone & Set Up Environment
Ensure you have Python 3.11+ installed.
```bash
# Clone the repository and enter directory
cd ShutterIQ

# Create a virtual environment
python -m venv .venv

# Activate the environment (macOS/Linux)
source .venv/bin/activate

# Activate the environment (Windows)
.venv\Scripts\activate

# Install all pinned dependencies
pip install -r requirements.txt
```

### 5.2. Run the Test Suite
Validate the entire ML pipeline, API gateway routes, and helper formatting utilities:
```bash
pytest -v
```

### 5.3. Launch the API Backend Server
Start the FastAPI server (preloads trained models into memory on startup):
```bash
.venv/bin/uvicorn api.main:app --port 8000
```
Interactive Swagger documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 5.4. Launch the Streamlit Frontend Dashboard
In a new terminal window (with the virtual environment activated), start the user-facing app:
```bash
.venv/bin/streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 6. Streamlit Dashboard Screenshots

Use the following section to drop in screenshots of the user interface after running the Streamlit app:

1. **Scheduling Input Dashboard:**
   *(Drop screenshot showing the sidebar connection hub and parameters entry form)*
   `![Scheduling Form Input](docs/form_input.png)`

2. **Duration SHAP Attributions Chart:**
   *(Drop screenshot showing the Plotly horizontal SHAP feature contributions chart)*
   `![Duration SHAP Chart](docs/duration_shap.png)`

3. **Dynamic Revenue Elasticity curve:**
   *(Drop screenshot showing the price optimization expected revenue curve)*
   `![Price optimization curve](docs/price_curve.png)`

4. **Ranked Booking Slots Timeline:**
   *(Drop screenshot showing the golden hour and weather-scored time slots)*
   `![Ranked Slots Table](docs/slots_ranking.png)`

---

## 7. Project Structure

```text
ShutterIQ/
├── api/
│   ├── main.py                     # FastAPI application endpoints
│   └── schemas.py                  # Pydantic request/response schemas
├── app/
│   ├── helpers.py                  # Extracted display formatting and SHAP helper functions
│   └── streamlit_app.py            # Streamlit dashboard interface
├── data/
│   └── bookings.csv                # Generated synthetic bookings dataset
├── models/
│   ├── duration_model.py           # XGBoost shoot duration & buffer training
│   ├── pricing_model.py            # Dynamic WTP elasticity pricing engine
│   └── weather_optimizer.py        # Astronomical solar & weather slots ranker
├── reports/
│   └── duration_model_shap.png     # Saved offline SHAP attribution plot
├── simulate/
│   └── generate_bookings.py        # Booking request simulator and data generator
├── tests/
│   ├── test_api.py                 # FastAPI client and validation tests
│   ├── test_duration_model.py      # Duration pipeline and buffer bounds checks
│   ├── test_generate_bookings.py   # Dataset validation and simulation checks
│   ├── test_pricing_model.py       # Elasticity and WTP recovery checks
│   ├── test_streamlit_app.py       # Helper functions and formatter checks
│   └── test_weather_optimizer.py   # Golden hour and fallback checks
├── CHANGELOG.md                    # Release and development stage history
├── LICENSE                         # MIT License file
├── requirements.txt                # Pinned python dependency list
└── README.md                       # Cohesive project documentation
```

---

## 8. Development & Stage History

For a complete breakdown of features added during each stage of ShutterIQ's implementation, refer to the [CHANGELOG.md](CHANGELOG.md) file.
