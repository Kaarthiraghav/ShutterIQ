"""
ShutterIQ API - Pydantic Schemas

Defines request and response validation schemas for FastAPI endpoints.
All schemas perform input validation on categorical bounds, range limits,
and coordinate bounds.
"""

from datetime import date
from typing import List
from pydantic import BaseModel, Field, field_validator


# --- Duration Engine Schemas ---
class DurationRequest(BaseModel):
    shoot_type: str = Field(..., description="Photography service category")
    num_people: int = Field(..., description="Number of people in the shoot group")
    location_type: str = Field(..., description="Indoor or outdoor setting context")
    lighting_setup: str = Field(..., description="Lighting gear configuration used")
    photographer_experience_years: int = Field(..., description="Photographer experience in years")

    @field_validator("shoot_type")
    @classmethod
    def validate_shoot_type(cls, v: str) -> str:
        valid = ["portrait", "graduation", "nature", "event", "wedding"]
        if v not in valid:
            raise ValueError(f"shoot_type must be one of {valid}")
        return v

    @field_validator("location_type")
    @classmethod
    def validate_location_type(cls, v: str) -> str:
        valid = ["indoor", "outdoor"]
        if v not in valid:
            raise ValueError(f"location_type must be one of {valid}")
        return v

    @field_validator("lighting_setup")
    @classmethod
    def validate_lighting_setup(cls, v: str) -> str:
        valid = ["studio_lights", "speedlight", "natural_light", "none"]
        if v not in valid:
            raise ValueError(f"lighting_setup must be one of {valid}")
        return v

    @field_validator("num_people")
    @classmethod
    def validate_num_people(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_people must be at least 1")
        return v

    @field_validator("photographer_experience_years")
    @classmethod
    def validate_experience(cls, v: int) -> int:
        if not (1 <= v <= 20):
            raise ValueError("photographer_experience_years must be in range [1, 20]")
        return v


class DurationResponse(BaseModel):
    predicted_duration_minutes: float = Field(..., description="Forecasted duration in minutes")
    recommended_buffer_minutes: int = Field(..., description="Recommended buffer protect window in minutes")


# --- Pricing Engine Schemas ---
class PriceRequest(BaseModel):
    shoot_type: str = Field(..., description="Photography service category")
    season: str = Field(..., description="Season classification (e.g. regular_season)")
    day_of_week: str = Field(..., description="Day of the week (e.g. Monday)")
    lead_time_days: int = Field(..., description="Days between booking date and shoot date")
    photographer_experience_years: int = Field(..., description="Photographer experience in years")
    num_people: int = Field(..., description="Number of participants in the shoot")

    @field_validator("shoot_type")
    @classmethod
    def validate_shoot_type(cls, v: str) -> str:
        valid = ["portrait", "graduation", "nature", "event", "wedding"]
        if v not in valid:
            raise ValueError(f"shoot_type must be one of {valid}")
        return v

    @field_validator("season")
    @classmethod
    def validate_season(cls, v: str) -> str:
        valid = ["regular_season", "wedding_season", "graduation_season"]
        if v not in valid:
            raise ValueError(f"season must be one of {valid}")
        return v

    @field_validator("day_of_week")
    @classmethod
    def validate_day_of_week(cls, v: str) -> str:
        valid = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if v not in valid:
            raise ValueError(f"day_of_week must be one of {valid}")
        return v

    @field_validator("lead_time_days")
    @classmethod
    def validate_lead_time(cls, v: int) -> int:
        if v < 1:
            raise ValueError("lead_time_days must be at least 1")
        return v

    @field_validator("photographer_experience_years")
    @classmethod
    def validate_experience(cls, v: int) -> int:
        if not (1 <= v <= 20):
            raise ValueError("photographer_experience_years must be in range [1, 20]")
        return v

    @field_validator("num_people")
    @classmethod
    def validate_num_people(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_people must be at least 1")
        return v


class PriceResponse(BaseModel):
    recommended_price: float = Field(..., description="Optimized revenue-maximizing quote price in USD")
    expected_revenue: float = Field(..., description="Expected dynamic revenue (price * P(Accept)) in USD")


# --- Weather Optimizer Schemas ---
class SlotRequest(BaseModel):
    latitude: float = Field(..., description="Location latitude")
    longitude: float = Field(..., description="Location longitude")
    start_date: str = Field(..., description="Start date of range (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date of range (YYYY-MM-DD)")
    timezone: str = Field("Asia/Colombo", description="Local timezone identifier")

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError("latitude must be in range [-90.0, 90.0]")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError("longitude must be in range [-180.0, 180.0]")
        return v

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_dates(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except Exception:
            raise ValueError("date must be in YYYY-MM-DD format")
        return v


class SlotDetail(BaseModel):
    start_time: str = Field(..., description="Local timezone-aware start datetime string")
    end_time: str = Field(..., description="Local timezone-aware end datetime string")
    score: float = Field(..., description="Composite slot quality score in [0, 1]")
    lighting_score: float = Field(..., description="Astronomical solar light quality score")
    cloud_score: float = Field(..., description="Diffused sky lighting quality score")
    precip_score: float = Field(..., description="Precipitation safety score")
    cloud_cover_percent: float = Field(..., description="Cloud cover percentage")
    precip_prob_percent: float = Field(..., description="Precipitation probability percentage")
    temperature_c: float = Field(..., description="Temperature in degrees Celsius")
    weather_available: bool = Field(..., description="True if meteorological forecast data was available")


class SlotResponse(BaseModel):
    slots: List[SlotDetail] = Field(..., description="Ranked list of best slots, descending by score")


# --- Combined Booking Optimizer Schemas ---
class ScheduleSuggestionRequest(BaseModel):
    shoot_type: str = Field(..., description="Photography service category")
    num_people: int = Field(..., description="Number of group participants")
    location_type: str = Field(..., description="Setting context (indoor or outdoor)")
    lighting_setup: str = Field(..., description="Lighting gear configuration used")
    photographer_experience_years: int = Field(..., description="Photographer experience in years")
    season: str = Field(..., description="Season category classification")
    day_of_week: str = Field(..., description="Booking day of week")
    lead_time_days: int = Field(..., description="Days between booking date and shoot date")
    latitude: float = Field(..., description="Coordinate location latitude")
    longitude: float = Field(..., description="Coordinate location longitude")
    start_date: str = Field(..., description="Date range search start (YYYY-MM-DD)")
    end_date: str = Field(..., description="Date range search end (YYYY-MM-DD)")
    timezone: str = Field("Asia/Colombo", description="Local timezone identifier")

    @field_validator("shoot_type")
    @classmethod
    def validate_shoot_type(cls, v: str) -> str:
        valid = ["portrait", "graduation", "nature", "event", "wedding"]
        if v not in valid:
            raise ValueError(f"shoot_type must be one of {valid}")
        return v

    @field_validator("location_type")
    @classmethod
    def validate_location_type(cls, v: str) -> str:
        valid = ["indoor", "outdoor"]
        if v not in valid:
            raise ValueError(f"location_type must be one of {valid}")
        return v

    @field_validator("lighting_setup")
    @classmethod
    def validate_lighting_setup(cls, v: str) -> str:
        valid = ["studio_lights", "speedlight", "natural_light", "none"]
        if v not in valid:
            raise ValueError(f"lighting_setup must be one of {valid}")
        return v

    @field_validator("season")
    @classmethod
    def validate_season(cls, v: str) -> str:
        valid = ["regular_season", "wedding_season", "graduation_season"]
        if v not in valid:
            raise ValueError(f"season must be one of {valid}")
        return v

    @field_validator("day_of_week")
    @classmethod
    def validate_day_of_week(cls, v: str) -> str:
        valid = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if v not in valid:
            raise ValueError(f"day_of_week must be one of {valid}")
        return v

    @field_validator("num_people")
    @classmethod
    def validate_num_people(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_people must be at least 1")
        return v

    @field_validator("photographer_experience_years")
    @classmethod
    def validate_experience(cls, v: int) -> int:
        if not (1 <= v <= 20):
            raise ValueError("photographer_experience_years must be in range [1, 20]")
        return v

    @field_validator("lead_time_days")
    @classmethod
    def validate_lead_time(cls, v: int) -> int:
        if v < 1:
            raise ValueError("lead_time_days must be at least 1")
        return v

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError("latitude must be in range [-90.0, 90.0]")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError("longitude must be in range [-180.0, 180.0]")
        return v

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_dates(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except Exception:
            raise ValueError("date must be in YYYY-MM-DD format")
        return v


class ScheduleSuggestionResponse(BaseModel):
    duration_prediction: DurationResponse = Field(..., description="Duration predictor outputs")
    price_recommendation: PriceResponse = Field(..., description="Dynamic price engine outputs")
    best_slots: List[SlotDetail] = Field(..., description="Ranked outdoor slots (empty list if indoor)")
