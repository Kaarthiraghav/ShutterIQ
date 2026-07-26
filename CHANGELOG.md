# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-07-26
### Added
- Golden hour and weather scheduling optimizer (`models/weather_optimizer.py`) for ranking outdoor booking slots.
- Astronomical sunrise, sunset, golden hour, and blue hour boundary calculations using `astral`.
- Weather forecast API queries integration with the free, public Open-Meteo endpoint.
- Composite scoring logic with a multiplicative rain veto to penalize precipitation and a photography diffused-light cloud cover curve (peaking at 30%).
- Graceful API failure degradation path, automatically falling back to solar-only calculations when network requests fail or time out.
- Optimization unit test suite (`tests/test_weather_optimizer.py`) including mock response testing, rain veto validation, and timeout fallback checks.
- Comprehensive documentation added to the README and CHANGELOG.

## [0.3.0] - 2026-07-26
### Added
- Dynamic pricing engine (`models/pricing_model.py`) using price elasticity of demand to recommend yield-maximizing quote rates.
- Interaction features between price and shoot types to allow category-specific elasticity estimation.
- Exogenous shifter control logic to eliminate multicollinearity and isolate price sensitivity coefficients.
- Log-price regression baseline model, guaranteeing positive price outputs and significantly reducing baseline MAE.
- Analytical willingness-to-pay (WTP) recovery calculation and ground-truth evaluation.
- Pricing test suite (`tests/test_pricing_model.py`) validating model training, prediction boundaries, expected revenue optimization, and recovery accuracy.
- Detailed README section outlining dynamic pricing framing, causal constraints, and validation results.

## [0.2.0] - 2026-07-25
### Added
- XGBoost shoot duration model pipeline (`models/duration_model.py`)
- Baseline Linear Regression model for comparative evaluation
- Category-specific scheduling buffer calculation based on prediction residual standard deviations
- SHAP feature explanation pipeline (`models/explain_duration.py`) and summary plot visualization
- Test suite verifying training correctness, positive predictions, and buffer scaling bounds
- Updated README with model performance metrics and SHAP analysis plot

## [0.1.0] - 2026-07-25
### Added
- Repository scaffolding (flat root-level directory structure)
- Virtual environment setup and project dependencies (`requirements.txt`)
- Synthetic data generation engine (`simulate/generate_bookings.py`)
- Core test suite (`tests/test_generate_bookings.py`)
- Project README explaining core components and setup
