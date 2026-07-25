# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-07-25
### Added
- XGBoost shoot duration model pipeline (`shutteriq/models/duration_model.py`)
- Baseline Linear Regression model for comparative evaluation
- Category-specific scheduling buffer calculation based on prediction residual standard deviations
- SHAP feature explanation pipeline (`shutteriq/models/explain_duration.py`) and summary plot visualization
- Test suite verifying training correctness, positive predictions, and buffer scaling bounds
- Updated README with model performance metrics and SHAP analysis plot

## [0.1.0] - 2026-07-25
### Added
- Repository scaffolding (`shutteriq/` directory structure)
- Virtual environment setup and project dependencies (`requirements.txt`)
- Synthetic data generation engine (`shutteriq/simulate/generate_bookings.py`)
- Core test suite (`shutteriq/tests/test_generate_bookings.py`)
- Project README explaining core components and setup
