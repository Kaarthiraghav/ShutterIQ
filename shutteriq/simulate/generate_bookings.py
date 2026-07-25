"""
ShutterIQ - Booking Data Generator

This module implements a synthetic data generator for photography booking records.
It models realistic relationships between shoot types, group sizes, lighting setups,
seasons, photographer experience, pricing, and client acceptance (willingness-to-pay).

The generated dataset is designed to support three future ML/data components:
1. Shoot Duration Predictor (Regression)
2. Dynamic Pricing Engine (Price Elasticity & Classification)
3. Golden Hour & Weather Optimizer (Scheduling)
"""

import os
import uuid
import argparse
import calendar
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# =====================================================================
# CONFIGURABLE CONSTANTS (Distribution & Pricing Assumptions)
# =====================================================================
# These constants define the mathematical parameters of the simulation.
# Adjusting these values allows fine-tuning of the pricing elasticity,
# duration trends, and seasonal spikes.

# Base duration in minutes. Represents the median duration of a simple shoot of each type.
# - Portrait: Quick sessions, minimal overhead.
# - Graduation: Multi-setup, group shots.
# - Nature: Requires travel and scouting.
# - Event: Birthdays/conferences, typically half-day.
# - Wedding: High stake, multi-location, full-day event.
BASE_DURATIONS = {
    "portrait": 45.0,
    "graduation": 60.0,
    "nature": 120.0,
    "event": 180.0,
    "wedding": 360.0
}

# Base pricing in USD. Baseline commercial rates for standard services.
BASE_PRICES = {
    "portrait": 150.0,
    "graduation": 200.0,
    "nature": 250.0,
    "event": 600.0,
    "wedding": 2000.0
}

# Baseline Willingness-to-Pay (WTP) in USD.
# Reflects the target customer profile budget. Slightly higher than base prices,
# giving a healthy base booking rate of ~60-70% when no premium is applied.
BASE_WTP = {
    "portrait": 170.0,
    "graduation": 220.0,
    "nature": 280.0,
    "event": 650.0,
    "wedding": 2200.0
}

# Duration formula scaling factors:
# - ALPHA: Controls the logarithmic scaling of duration as group size increases.
# - BETA: Outdoor premium factor (setup, transit, natural light delays).
DURATION_ALPHA = 0.25
DURATION_BETA = 0.15

# Photographer experience scaling parameters (per year of experience):
# - PRICE_EXP_COEFF: Experienced photographers charge 5% premium per year of experience.
# - WTP_EXP_COEFF: Clients value experience, but are slightly less sensitive than asking price (4% per year).
PRICE_EXP_COEFF = 0.05
WTP_EXP_COEFF = 0.04

# Seasonality settings:
# Sri Lankan grad season peaks in July/August.
# Sri Lankan wedding season peaks in May/June and Nov/Dec.
SEASON_MULTIPLIERS = {
    "wedding_season": 1.35,       # +35% price/WTP during peak wedding months
    "graduation_season": 1.25,    # +25% price/WTP during peak graduation months
    "regular_season": 1.0
}

# Day of week price modifiers. Weekends are premium shooting days.
DAY_OF_WEEK_MULTIPLIERS = {
    "Monday": 1.0,
    "Tuesday": 1.0,
    "Wednesday": 1.0,
    "Thursday": 1.0,
    "Friday": 1.10,
    "Saturday": 1.20,
    "Sunday": 1.20
}

# Lead time price modifiers. Last-minute bookings command convenience premiums,
# while early-bird bookings receive modest planning discounts.
LEAD_TIME_MULTIPLIERS = {
    "last_minute": 1.30,  # < 7 days
    "mid_range": 1.15,    # 7-14 days
    "standard": 1.0,      # 14-60 days
    "early_bird": 0.90    # > 60 days
}

# Price sensitivity factor (k) in the logistic acceptance function:
# P(Accept) = 1 / (1 + exp(k * (price / WTP - 1)))
# Higher k means a steeper drop-off in acceptance once price exceeds willingness-to-pay.
PRICE_ELASTICITY_K = 7.0


# =====================================================================
# DATA SAMPLING HELPERS
# =====================================================================

def sample_requested_datetime(shoot_type: str, start_date: datetime, end_date: datetime) -> datetime:
    """
    Samples a realistic requested datetime reflecting Sri Lankan seasonality
    and shoot-type-specific hour-of-day distributions.
    """
    months = list(range(1, 13))
    
    # Define monthly weights for seasonality to match real-world peaks
    if shoot_type == "wedding":
        # Peaks in May, June, November, December
        weights = [1 if m not in [5, 6, 11, 12] else 4 for m in months]
    elif shoot_type == "graduation":
        # Peaks in July, August
        weights = [1 if m not in [7, 8] else 6 for m in months]
    else:
        weights = [1] * 12
        
    weights = np.array(weights) / sum(weights)
    month = np.random.choice(months, p=weights)
    
    # Select year in range
    year = np.random.choice(list(range(start_date.year, end_date.year + 1)))
    
    # Handle valid day matching calendar limits
    _, num_days = calendar.monthrange(year, month)
    day = np.random.randint(1, num_days + 1)
    
    # Select realistic hours of the day
    if shoot_type in ["nature", "portrait", "graduation"]:
        # Portrates, grads, and nature shoots favor golden/daylight hours (7 AM - 6 PM)
        # Golden hours (7-9 AM, 4-6 PM) get higher weights
        hours = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        hour_weights = [3, 3, 1, 1, 1, 1, 1, 1, 2, 3, 3, 2]
        hour_weights = np.array(hour_weights) / sum(hour_weights)
        hour = np.random.choice(hours, p=hour_weights)
    else:
        # Weddings and general events span longer afternoon/evening blocks
        hour = np.random.randint(9, 21)
        
    minute = np.random.choice([0, 15, 30, 45])
    
    dt = datetime(year, month, day, hour, minute)
    
    # Clamp to specified date range boundary
    if dt < start_date:
        dt = dt + timedelta(days=365)
    elif dt > end_date:
        dt = dt - timedelta(days=365)
        
    return dt


def get_season(dt: datetime) -> str:
    """
    Categorizes the date's season based on Sri Lankan holiday/festive patterns.
    """
    month = dt.month
    if month in [5, 6, 11, 12]:
        return "wedding_season"
    elif month in [7, 8]:
        return "graduation_season"
    else:
        return "regular_season"


def sample_lead_time(shoot_type: str) -> int:
    """
    Samples booking lead time. Weddings require long planning phases, whereas
    portraits and nature shoots are booked much closer to the requested date.
    """
    if shoot_type == "wedding":
        # Average 3 months lead time, range [14, 180] days
        val = np.random.normal(90, 30)
        return int(np.clip(val, 14, 180))
    elif shoot_type == "event":
        # Average 1 month, range [5, 90] days
        val = np.random.normal(30, 15)
        return int(np.clip(val, 5, 90))
    elif shoot_type == "graduation":
        # Average 20 days, range [3, 60] days
        val = np.random.normal(20, 10)
        return int(np.clip(val, 3, 60))
    elif shoot_type == "portrait":
        # Average 10 days, range [2, 30] days
        val = np.random.normal(10, 5)
        return int(np.clip(val, 2, 30))
    else:  # nature
        # Nature shoots are highly weather dependent, booked short notice: [1, 21] days
        val = np.random.normal(7, 4)
        return int(np.clip(val, 1, 21))


def sample_num_people(shoot_type: str) -> int:
    """
    Samples group sizes. Weddings and events represent crowd bookings,
    while portraits/nature shoots focus on individuals or couples.
    """
    if shoot_type == "portrait":
        return int(np.random.randint(1, 6))
    elif shoot_type == "graduation":
        return int(np.random.randint(1, 9))
    elif shoot_type == "wedding":
        # Standard wedding guest list: average 150, range [30, 400]
        val = np.random.normal(150, 45)
        return int(np.clip(val, 30, 400))
    elif shoot_type == "event":
        # Event guest lists: average 60, range [10, 250]
        val = np.random.normal(60, 30)
        return int(np.clip(val, 10, 250))
    else:  # nature
        # Landscape or macro photography involves 1-2 people
        return int(np.random.choice([1, 2], p=[0.8, 0.2]))


def sample_location_type(shoot_type: str) -> str:
    """
    Determines if a shoot is indoor or outdoor based on typical format likelihoods.
    """
    if shoot_type == "portrait":
        return np.random.choice(["indoor", "outdoor"], p=[0.6, 0.4])
    elif shoot_type == "graduation":
        return np.random.choice(["indoor", "outdoor"], p=[0.2, 0.8])
    elif shoot_type == "wedding":
        return np.random.choice(["indoor", "outdoor"], p=[0.4, 0.6])
    elif shoot_type == "event":
        return np.random.choice(["indoor", "outdoor"], p=[0.8, 0.2])
    else:  # nature
        return "outdoor"


def sample_lighting_setup(location_type: str) -> str:
    """
    Suggests suitable lighting hardware depending on indoor/outdoor context.
    """
    if location_type == "outdoor":
        return np.random.choice(["natural_light", "speedlight", "none"], p=[0.75, 0.15, 0.10])
    else:  # indoor
        return np.random.choice(["studio_lights", "speedlight", "none"], p=[0.65, 0.25, 0.10])


def get_lead_time_category(lead_time: int) -> str:
    """
    Classifies lead time days into discrete pricing tiers.
    """
    if lead_time < 7:
        return "last_minute"
    elif lead_time < 14:
        return "mid_range"
    elif lead_time <= 60:
        return "standard"
    else:
        return "early_bird"


# =====================================================================
# CALCULATION LOGIC (Non-linear & Compounding Models)
# =====================================================================

def calculate_duration(shoot_type: str, num_people: int, location_type: str) -> int:
    """
    Computes shoot duration.
    Models duration as a non-linear function:
      Duration = Base * (1 + alpha * log(num_people)) * (1 + beta * is_outdoor) + noise
      
    - Logarithmic scaling for group size models the efficiency curve of staging larger numbers.
    - Outdoor premium factors in environmental transitions and weather delays.
    - Log-normal multiplicative noise ensures non-negative fluctuations.
    """
    base = BASE_DURATIONS[shoot_type]
    is_outdoor = 1.0 if location_type == "outdoor" else 0.0
    
    # Logarithmic group scaling (base-e) ensures incremental slowdown diminishes for massive groups
    group_scaling = 1.0 + DURATION_ALPHA * np.log(num_people)
    outdoor_scaling = 1.0 + DURATION_BETA * is_outdoor
    
    modeled_duration = base * group_scaling * outdoor_scaling
    
    # Log-normal noise prevents negative duration values (stddev ~ 8% of target duration)
    noise = np.random.lognormal(mean=0.0, sigma=0.08)
    actual_duration = modeled_duration * noise
    
    # Ensure a logical floor of 15 minutes, round to nearest minute for schedule formatting
    return int(max(15, round(actual_duration)))


def calculate_price_and_acceptance(
    shoot_type: str,
    season: str,
    day_of_week: str,
    lead_time_days: int,
    experience_years: int,
    num_people: int
) -> tuple[float, float, bool]:
    """
    Calculates quote price and customer acceptance status.
    
    1. Quoted Price: Compounding factors scale base service rate:
       Price = BasePrice * SeasonMult * DayOfWeekMult * LeadTimeMult * ExperienceMult
       Plus a +/-5% negotiation random variance, rounded to the nearest $5.
       
    2. Willingness-to-Pay (WTP): Customer's baseline value ceiling:
       WTP = BaseWTP * SeasonMult * ExperienceWTPMult * GroupWTPMult * ClientBudgetNoise
       Where ClientBudgetNoise is log-normal (stddev 12%) reflecting varied customer financial capacity.
       
    3. Acceptance Rate: Modeled using a logistic probability curve:
       P(Accept) = 1 / (1 + exp(k * (Price / WTP - 1)))
       The boolean status is sampled from this probability (Bernoulli draw).
    """
    # 1. Base Prices
    base_price = BASE_PRICES[shoot_type]
    base_wtp = BASE_WTP[shoot_type]
    
    # 2. Multipliers
    season_mult = SEASON_MULTIPLIERS[season]
    day_mult = DAY_OF_WEEK_MULTIPLIERS[day_of_week]
    
    lead_category = get_lead_time_category(lead_time_days)
    lead_mult = LEAD_TIME_MULTIPLIERS[lead_category]
    
    price_exp_mult = 1.0 + PRICE_EXP_COEFF * experience_years
    wtp_exp_mult = 1.0 + WTP_EXP_COEFF * experience_years
    
    # Clients anticipate slightly higher WTP for larger group bookings
    group_wtp_mult = 1.0 + 0.08 * np.log(num_people)
    
    # 3. Compute Final Quoted Price (with slight commercial negotiation variation)
    quoted_price = base_price * season_mult * day_mult * lead_mult * price_exp_mult
    quoted_price *= np.random.uniform(0.95, 1.05)
    quoted_price = float(np.round(quoted_price / 5.0) * 5.0)  # Commercial clean rounding
    
    # 4. Compute Customer Willingness-to-Pay (WTP)
    client_noise = np.random.lognormal(mean=0.0, sigma=0.12)
    wtp = base_wtp * season_mult * wtp_exp_mult * group_wtp_mult * client_noise
    wtp = float(np.round(wtp / 5.0) * 5.0)
    
    # 5. Determine Acceptance using Logistic Elasticity Function
    price_ratio = quoted_price / wtp
    prob_accept = 1.0 / (1.0 + np.exp(PRICE_ELASTICITY_K * (price_ratio - 1.0)))
    booking_accepted = bool(np.random.rand() < prob_accept)
    
    return quoted_price, wtp, booking_accepted


# =====================================================================
# MAIN SIMULATOR INTERFACE
# =====================================================================

def generate_dataset(num_records: int, seed: int = 42) -> pd.DataFrame:
    """
    Generates a complete dataset of synthetic photography booking records.
    """
    np.random.seed(seed)
    
    # Define timeframe: past 2 years to future 6 months
    today = datetime.now()
    start_date = today - timedelta(days=730)
    end_date = today + timedelta(days=180)
    
    records = []
    shoot_types = ["portrait", "event", "wedding", "graduation", "nature"]
    shoot_weights = [0.35, 0.20, 0.15, 0.20, 0.10]
    
    for _ in range(num_records):
        # 1. Sample basic attributes
        booking_id = str(uuid.uuid4())
        shoot_type = np.random.choice(shoot_types, p=shoot_weights)
        num_people = sample_num_people(shoot_type)
        location_type = sample_location_type(shoot_type)
        lighting_setup = sample_lighting_setup(location_type)
        photographer_experience_years = int(np.random.randint(1, 21))
        lead_time_days = sample_lead_time(shoot_type)
        
        # 2. Sample requested datetime & derive calendar properties
        requested_datetime = sample_requested_datetime(shoot_type, start_date, end_date)
        day_of_week = requested_datetime.strftime("%A")
        season = get_season(requested_datetime)
        
        # 3. Calculate target outputs (duration & price acceptance)
        actual_duration_minutes = calculate_duration(shoot_type, num_people, location_type)
        quoted_price, true_wtp, booking_accepted = calculate_price_and_acceptance(
            shoot_type=shoot_type,
            season=season,
            day_of_week=day_of_week,
            lead_time_days=lead_time_days,
            experience_years=photographer_experience_years,
            num_people=num_people
        )
        
        records.append({
            "booking_id": booking_id,
            "shoot_type": shoot_type,
            "num_people": num_people,
            "location_type": location_type,
            "lighting_setup": lighting_setup,
            "photographer_experience_years": photographer_experience_years,
            "lead_time_days": lead_time_days,
            "day_of_week": day_of_week,
            "season": season,
            "requested_datetime": requested_datetime.isoformat(),
            "actual_duration_minutes": actual_duration_minutes,
            "quoted_price": quoted_price,
            "true_wtp": true_wtp,  # Included as ground truth validation baseline
            "booking_accepted": booking_accepted
        })
        
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic booking records for ShutterIQ.")
    parser.add_argument("--num-records", type=int, default=1000, help="Number of records to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="shutteriq/data/bookings.csv", help="Path to write CSV file.")
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_records} synthetic booking records (seed={args.seed})...")
    df = generate_dataset(args.num_records, args.seed)
    
    # Ensure data folder exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df.to_csv(args.output, index=False)
    print(f"Successfully generated dataset at {args.output}")
