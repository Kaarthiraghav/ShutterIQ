"""
ShutterIQ FastAPI Application Layer

Exposes machine learning models and heuristics for photographers behind REST endpoints:
- POST /predict-duration
- POST /recommend-price
- POST /best-slots
- POST /schedule-suggestion
- GET /health
"""

import sys
import os
from contextlib import asynccontextmanager
from datetime import date
from fastapi import FastAPI

# Add parent directory to sys.path to allow clean imports from models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.duration_model import (
    predict_duration_with_buffer,
    _get_artifacts as load_duration_artifacts
)
from models.pricing_model import (
    recommend_price_and_expected_revenue,
    _get_artifacts as load_pricing_artifacts
)
from models.weather_optimizer import score_slots

from api.schemas import (
    DurationRequest,
    DurationResponse,
    PriceRequest,
    PriceResponse,
    SlotRequest,
    SlotDetail,
    SlotResponse,
    ScheduleSuggestionRequest,
    ScheduleSuggestionResponse
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler. Preloads pickled ML models on startup
    to prevent first-request latency.
    """
    print("Starting up ShutterIQ API: preloading trained model pipelines...")
    load_duration_artifacts()
    load_pricing_artifacts()
    print("Model preloading complete. ShutterIQ API ready.")
    yield
    print("Shutting down ShutterIQ API.")


app = FastAPI(
    title="ShutterIQ API",
    description="Intelligent scheduling and pricing assistant for photographers",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """
    API Health Probe Endpoint.
    """
    return {"status": "OK"}


@app.post("/predict-duration", response_model=DurationResponse)
async def predict_duration(req: DurationRequest):
    """
    Forecasts actual shoot duration and recommends a 95% protection buffer.
    """
    pred, buf = predict_duration_with_buffer(
        shoot_type=req.shoot_type,
        num_people=req.num_people,
        location_type=req.location_type,
        lighting_setup=req.lighting_setup,
        photographer_experience_years=req.photographer_experience_years
    )
    return DurationResponse(
        predicted_duration_minutes=pred,
        recommended_buffer_minutes=buf
    )


@app.post("/recommend-price", response_model=PriceResponse)
async def recommend_price(req: PriceRequest):
    """
    Recommends a yield-maximizing quote price based on request elasticity factors.
    """
    booking_features = req.model_dump()
    rec_price, exp_revenue = recommend_price_and_expected_revenue(booking_features)
    return PriceResponse(
        recommended_price=rec_price,
        expected_revenue=exp_revenue
    )


@app.post("/best-slots", response_model=SlotResponse)
async def best_slots(req: SlotRequest):
    """
    Retrieves and ranks outdoor booking slots by combining solar and weather quality.
    """
    start_dt = date.fromisoformat(req.start_date)
    end_dt = date.fromisoformat(req.end_date)
    
    slots = score_slots(
        latitude=req.latitude,
        longitude=req.longitude,
        timezone_str=req.timezone,
        start_date=start_dt,
        end_date=end_dt
    )
    
    slot_details = [
        SlotDetail(
            start_time=s["start_time"],
            end_time=s["end_time"],
            score=s["score"],
            lighting_score=s["lighting_score"],
            cloud_score=s["cloud_score"],
            precip_score=s["precip_score"],
            cloud_cover_percent=s["cloud_cover_percent"],
            precip_prob_percent=s["precip_prob_percent"],
            temperature_c=s["temperature_c"],
            weather_available=s["weather_available"]
        ) for s in slots
    ]
    return SlotResponse(slots=slot_details)


@app.post("/schedule-suggestion", response_model=ScheduleSuggestionResponse)
async def schedule_suggestion(req: ScheduleSuggestionRequest):
    """
    Combined scheduling optimization endpoint. Runs predicted duration,
    optimized price, and (if outdoor) ranked solar/weather time slots together.
    """
    # 1. Predict Duration
    pred, buf = predict_duration_with_buffer(
        shoot_type=req.shoot_type,
        num_people=req.num_people,
        location_type=req.location_type,
        lighting_setup=req.lighting_setup,
        photographer_experience_years=req.photographer_experience_years
    )
    duration_res = DurationResponse(
        predicted_duration_minutes=pred,
        recommended_buffer_minutes=buf
    )

    # 2. Recommend Price
    booking_features = {
        "shoot_type": req.shoot_type,
        "season": req.season,
        "day_of_week": req.day_of_week,
        "lead_time_days": req.lead_time_days,
        "photographer_experience_years": req.photographer_experience_years,
        "num_people": req.num_people
    }
    rec_price, exp_revenue = recommend_price_and_expected_revenue(booking_features)
    price_res = PriceResponse(
        recommended_price=rec_price,
        expected_revenue=exp_revenue
    )

    # 3. Solar & Weather Slots (only calculated if outdoor)
    slot_details = []
    if req.location_type == "outdoor":
        start_dt = date.fromisoformat(req.start_date)
        end_dt = date.fromisoformat(req.end_date)
        
        slots = score_slots(
            latitude=req.latitude,
            longitude=req.longitude,
            timezone_str=req.timezone,
            start_date=start_dt,
            end_date=end_dt
        )
        slot_details = [
            SlotDetail(
                start_time=s["start_time"],
                end_time=s["end_time"],
                score=s["score"],
                lighting_score=s["lighting_score"],
                cloud_score=s["cloud_score"],
                precip_score=s["precip_score"],
                cloud_cover_percent=s["cloud_cover_percent"],
                precip_prob_percent=s["precip_prob_percent"],
                temperature_c=s["temperature_c"],
                weather_available=s["weather_available"]
            ) for s in slots
        ]

    return ScheduleSuggestionResponse(
        duration_prediction=duration_res,
        price_recommendation=price_res,
        best_slots=slot_details
    )
