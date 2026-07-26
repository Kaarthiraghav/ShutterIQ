"""
ShutterIQ - Golden Hour & Weather Optimization Engine

This module calculates optimal booking slots for outdoor photography.
It combines:
1. Astronomical solar calculations (golden hour and blue hour) using the `astral` library.
2. Weather forecasts (cloud cover, precipitation probability, temperature) from the Open-Meteo API.

In the event of weather API failures, it degrades gracefully to golden-hour-only scoring.
"""

import sys
import argparse
import logging
import zoneinfo
from datetime import datetime, date, timedelta
import requests

# Configure logging to stderr for visibility in CLI runs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger("weather_optimizer")

# List of photography shoot types (unused directly here, but for documentation reference)
OUTDOOR_SHOOT_TYPES = ["portrait", "graduation", "nature", "wedding", "event"]


def get_solar_windows(latitude: float, longitude: float, timezone_str: str, target_date: date) -> dict:
    """
    Computes morning and evening golden hour and blue hour windows for a location and date.
    Returns a dictionary of (start, end) datetime tuples, or None if unavailable/errors out.
    """
    from astral import Observer
    from astral.sun import golden_hour, blue_hour, SunDirection

    tz = zoneinfo.ZoneInfo(timezone_str)
    observer = Observer(latitude=latitude, longitude=longitude)

    windows = {
        "morning_golden": None,
        "evening_golden": None,
        "morning_blue": None,
        "evening_blue": None
    }

    # Morning Golden Hour (Rising)
    try:
        windows["morning_golden"] = golden_hour(observer, date=target_date, direction=SunDirection.RISING, tzinfo=tz)
    except Exception as e:
        logger.debug(f"Could not compute morning golden hour for {target_date}: {e}")

    # Evening Golden Hour (Setting)
    try:
        windows["evening_golden"] = golden_hour(observer, date=target_date, direction=SunDirection.SETTING, tzinfo=tz)
    except Exception as e:
        logger.debug(f"Could not compute evening golden hour for {target_date}: {e}")

    # Morning Blue Hour (Rising)
    try:
        windows["morning_blue"] = blue_hour(observer, date=target_date, direction=SunDirection.RISING, tzinfo=tz)
    except Exception as e:
        logger.debug(f"Could not compute morning blue hour for {target_date}: {e}")

    # Evening Blue Hour (Setting)
    try:
        windows["evening_blue"] = blue_hour(observer, date=target_date, direction=SunDirection.SETTING, tzinfo=tz)
    except Exception as e:
        logger.debug(f"Could not compute evening blue hour for {target_date}: {e}")

    return windows


def get_datetime_distance_to_interval(dt: datetime, interval: tuple[datetime, datetime] | None) -> float:
    """
    Returns the absolute distance in minutes from dt to the interval.
    If dt is inside the interval, distance is 0.0.
    If interval is None, returns infinity.
    """
    if interval is None:
        return float("inf")
    start, end = interval
    if dt < start:
        return (start - dt).total_seconds() / 60.0
    elif dt > end:
        return (dt - end).total_seconds() / 60.0
    else:
        return 0.0


def calculate_lighting_score(dt_mid: datetime, solar_windows: dict) -> float:
    """
    Calculates a lighting quality score in [0, 1] based on proximity to solar windows.
    - Inside golden hour: 1.0
    - Inside blue hour: 0.8
    - Outside: Decays exponentially (half-life of ~83 mins, exp(-d/120)) centered around the closest window.
    """
    # 1. Proximity to golden hour windows
    d_m_gold = get_datetime_distance_to_interval(dt_mid, solar_windows["morning_golden"])
    d_e_gold = get_datetime_distance_to_interval(dt_mid, solar_windows["evening_golden"])
    d_gold = min(d_m_gold, d_e_gold)

    # 2. Proximity to blue hour windows
    d_m_blue = get_datetime_distance_to_interval(dt_mid, solar_windows["morning_blue"])
    d_e_blue = get_datetime_distance_to_interval(dt_mid, solar_windows["evening_blue"])
    d_blue = min(d_m_blue, d_e_blue)

    # 3. Apply exponential decay scores
    score_gold = 1.0 * np.exp(-d_gold / 120.0) if d_gold != float("inf") else 0.0
    score_blue = 0.8 * np.exp(-d_blue / 120.0) if d_blue != float("inf") else 0.0

    return float(max(score_gold, score_blue))


def fetch_weather_forecast(latitude: float, longitude: float, start_date: date, end_date: date) -> dict | None:
    """
    Fetches hourly weather forecast from the free Open-Meteo API.
    Returns parsed dictionary of hourly weather data or None if request fails.
    """
    # Build Open-Meteo URL
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation_probability,cloud_cover",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "auto"
    }

    try:
        logger.info(f"Fetching weather forecast from Open-Meteo for range [{start_date}, {end_date}]...")
        response = requests.get(url, params=params, timeout=5.0)
        if response.status_code != 200:
            logger.warning(f"Open-Meteo API returned status code {response.status_code}: {response.text}")
            return None
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch weather forecast due to exception: {e}")
        return None


def parse_weather_data(weather_json: dict | None, timezone_str: str) -> dict:
    """
    Parses Open-Meteo forecast JSON into a mapping keyed by timezone-aware datetime string.
    If weather_json is None, returns an empty mapping.
    """
    if not weather_json or "hourly" not in weather_json:
        return {}

    hourly = weather_json["hourly"]
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precips = hourly.get("precipitation_probability", [])
    clouds = hourly.get("cloud_cover", [])

    tz = zoneinfo.ZoneInfo(timezone_str)
    weather_map = {}

    for idx, time_str in enumerate(times):
        try:
            # Open-Meteo returns isoformat without timezone (e.g. 2026-07-26T07:00).
            # We localize it using the observer's target timezone.
            dt_naive = datetime.fromisoformat(time_str)
            dt_aware = dt_naive.replace(tzinfo=tz)
            
            weather_map[dt_aware] = {
                "temp": temps[idx] if idx < len(temps) else float("nan"),
                "precip_prob": precips[idx] if idx < len(precips) else 0.0,
                "cloud_cover": clouds[idx] if idx < len(clouds) else 0.0
            }
        except Exception as e:
            logger.debug(f"Failed to parse weather hour index {idx}: {e}")

    return weather_map


def calculate_cloud_score(cloud_cover: float) -> float:
    """
    Calculates cloud cover score in [0, 1].
    - Peak score (1.0) is at 30% cloud cover (beautiful diffused natural light).
    - Harsh direct sunlight (0% clouds) drops to 0.6.
    - Completely overcast skies (100% clouds) drops to 0.4.
    """
    if cloud_cover < 30.0:
        # Scale linearly from 0.6 (at 0%) to 1.0 (at 30%)
        return 0.6 + 0.4 * (cloud_cover / 30.0)
    else:
        # Scale linearly from 1.0 (at 30%) to 0.4 (at 100%)
        return 1.0 - 0.6 * ((cloud_cover - 30.0) / 70.0)


import numpy as np

def score_slots(
    latitude: float,
    longitude: float,
    timezone_str: str,
    start_date: date,
    end_date: date,
    day_start_hour: int = 7,
    day_end_hour: int = 18
) -> list[dict]:
    """
    Generates hourly slots across target dates and scores them combining solar and weather metrics.
    
    Scoring Formula (Option A):
      Score = (0.6 * S_light + 0.4 * S_cloud) * S_precip
      
    If weather is unavailable, degrades to:
      Score = S_light
    """
    tz = zoneinfo.ZoneInfo(timezone_str)
    
    # 1. Fetch weather and parse
    weather_json = fetch_weather_forecast(latitude, longitude, start_date, end_date)
    weather_map = parse_weather_data(weather_json, timezone_str)
    weather_available = len(weather_map) > 0

    if not weather_available:
        logger.warning("Degrading to Golden Hour-only scoring due to weather data unavailability.")

    # 2. Iterate dates in range
    num_days = (end_date - start_date).days + 1
    target_dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    slots = []
    
    for t_date in target_dates:
        # Calculate solar windows for the target date
        solar_windows = get_solar_windows(latitude, longitude, timezone_str, t_date)
        
        # Generate hourly slots (e.g. 7 AM to 6 PM)
        for hour in range(day_start_hour, day_end_hour):
            slot_start = datetime(t_date.year, t_date.month, t_date.day, hour, 0, tzinfo=tz)
            slot_end = slot_start + timedelta(hours=1)
            slot_mid = slot_start + timedelta(minutes=30)
            
            # Retrieve weather mapping. Because API times are top-of-the-hour,
            # we query using the slot's start time.
            weather_hour = weather_map.get(slot_start)
            
            # 3. Scoring
            # Golden hour/light score
            s_light = calculate_lighting_score(slot_mid, solar_windows)
            
            if weather_available and weather_hour is not None:
                # Weather scores
                cloud_cover = weather_hour["cloud_cover"]
                precip_prob = weather_hour["precip_prob"]
                temp = weather_hour["temp"]
                
                s_cloud = calculate_cloud_score(cloud_cover)
                s_precip = 1.0 - (precip_prob / 100.0)
                
                # Multiplicative Veto logic: heavy rain drops total score to 0.
                total_score = (0.6 * s_light + 0.4 * s_cloud) * s_precip
            else:
                # Fallback / degraded path
                cloud_cover = float("nan")
                precip_prob = float("nan")
                temp = float("nan")
                
                s_cloud = 1.0
                s_precip = 1.0
                total_score = s_light

            slots.append({
                "start_time": slot_start.isoformat(),
                "end_time": slot_end.isoformat(),
                "score": round(total_score, 4),
                "lighting_score": round(s_light, 4),
                "cloud_score": round(s_cloud, 4) if not np.isnan(cloud_cover) else 1.0,
                "precip_score": round(s_precip, 4) if not np.isnan(precip_prob) else 1.0,
                "cloud_cover_percent": cloud_cover,
                "precip_prob_percent": precip_prob,
                "temperature_c": temp,
                "weather_available": weather_available
            })

    # Sort slots by score descending
    slots.sort(key=lambda s: s["score"], reverse=True)
    return slots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend outdoor photo session slots.")
    parser.add_argument("--lat", type=float, default=6.9271, help="Latitude (default: Colombo)")
    parser.add_argument("--lon", type=float, default=79.8612, help="Longitude (default: Colombo)")
    parser.add_argument("--timezone", type=str, default="Asia/Colombo", help="Timezone identifier")
    parser.add_argument("--days", type=int, default=3, help="Forecast date range size in days")

    args = parser.parse_args()

    start = date.today()
    end = start + timedelta(days=args.days - 1)

    print(f"Running scheduling optimization for Location: ({args.lat}, {args.lon}) in {args.timezone}")
    print(f"Date range: {start} to {end}\n")

    results = score_slots(
        latitude=args.lat,
        longitude=args.lon,
        timezone_str=args.timezone,
        start_date=start,
        end_date=end
    )

    print(f"Ranked Booking Time Slots (Top 10):")
    print(f"{'Start Time':25s} | {'End Time':25s} | {'Score':6s} | {'Light':5s} | {'Cloud':5s} | {'Precip':5s} | {'Temp':5s} | {'Weather':7s}")
    print("-" * 115)
    for slot in results[:10]:
        t_str = f"{slot['temperature_c']:.1f}°C" if not np.isnan(slot['temperature_c']) else "N/A"
        avail_str = "Yes" if slot["weather_available"] else "No"
        print(f"{slot['start_time']:25s} | {slot['end_time']:25s} | {slot['score']:.4f} | {slot['lighting_score']:.3f} | {slot['cloud_score']:.3f} | {slot['precip_score']:.3f} | {t_str:5s} | {avail_str:7s}")
