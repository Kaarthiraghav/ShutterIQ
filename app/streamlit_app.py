"""
ShutterIQ - Interactive Scheduling & Pricing Assistant Demo

This Streamlit application serves as the user-facing portal.
It integrates with the FastAPI backend, provides inputs for scheduling request,
visualizes duration prediction SHAP factors, explains yield-maximizing pricing adjustments,
and displays ranked outdoor golden hour & weather-optimized booking slots.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

import sys
import os

# Add repository root to python search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.helpers import format_duration, get_weather_badge, explain_duration_factors

# Configure Page
st.set_page_config(
    page_title="ShutterIQ - Scheduling Assistant",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (clean, modern glassmorphism aesthetic)
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f8f9fa;
    }
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #1f392c 0%, #3e7356 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2b5c3f;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #495057;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================================
# SIDEBAR PANEL
# =====================================================================
st.sidebar.markdown("## 📸 ShutterIQ Control Center")

# API URL Configuration
api_base_url = st.sidebar.text_input(
    "API Gateway URL",
    value=st.sidebar.selectbox(
        "Preserved Hosts",
        options=["http://127.0.0.1:8000", "http://localhost:8000"]
    ),
    help="FastAPI server location. Set SHUTTERIQ_API_URL env variable to override defaults."
)

# Connection Health Check
api_online = False
try:
    health_resp = requests.get(f"{api_base_url}/health", timeout=1.5)
    if health_resp.status_code == 200 and health_resp.json().get("status") == "OK":
        api_online = True
except Exception:
    pass

if api_online:
    st.sidebar.success("🟢 API Server Connected")
else:
    st.sidebar.error("🔴 API Server Offline")
    st.sidebar.warning(
        "The ShutterIQ API backend is currently unreachable. Please make sure the FastAPI server is running:\n"
        "```bash\n"
        ".venv/bin/uvicorn api.main:app --port 8000\n"
        "```"
    )

# About Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Embedded Intelligent Models")
st.sidebar.markdown(
    "1. **Shoot Duration Predictor:** XGBoost Regressor trained on historical parameters with derived category-specific "
    "operational buffers (95% protection threshold).\n\n"
    "2. **Dynamic Pricing Engine:** Multiplicative log-space baseline pricing regression combined with interaction-terms "
    "Logistic dynamic elasticity modeling.\n\n"
    "3. **Golden Hour & Weather Optimizer:** Combines astronomical sunrise/sunset calculations with real-time "
    "Open-Meteo forecasts and diffused-light curves."
)


# =====================================================================
# MAIN WORKSPACE
# =====================================================================
st.markdown("<h1 class='main-title'>ShutterIQ Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Intelligent Shoot Scheduling and Expected Revenue Maximization Dashboard</p>", unsafe_allow_html=True)

# Main Form Container
with st.form("suggestion_form"):
    st.markdown("### 📝 Session Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        shoot_type = st.selectbox(
            "Service Type",
            options=["portrait", "graduation", "nature", "event", "wedding"],
            index=0
        )
        num_people = st.number_input(
            "Group Size (People)",
            min_value=1,
            value=2,
            step=1
        )
        photographer_experience_years = st.slider(
            "Photographer Experience (Years)",
            min_value=1,
            max_value=20,
            value=5
        )
        lighting_setup = st.selectbox(
            "Lighting Configuration",
            options=["natural_light", "studio_lights", "speedlight", "none"],
            index=0
        )
        location_type = st.selectbox(
            "Setting Location",
            options=["indoor", "outdoor"],
            index=1
        )

    with col2:
        season = st.selectbox(
            "Booking Season",
            options=["regular_season", "wedding_season", "graduation_season"],
            index=0
        )
        day_of_week = st.selectbox(
            "Day of the Week",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=5 # Default Saturday
        )
        lead_time_days = st.number_input(
            "Lead Time (Days)",
            min_value=1,
            value=14,
            step=1
        )
        
        # Display coordinate selection only if setting is outdoor
        st.markdown("**Outdoor Coordinates (Open-Meteo Forecast Query)**")
        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            latitude = st.number_input("Latitude", value=6.9271, format="%.4f")
        with coord_col2:
            longitude = st.number_input("Longitude", value=79.8612, format="%.4f")
            
        # Date selection range for weather optimizer
        start_date = st.date_input("Schedule Start Date", value=date.today())
        end_date = st.date_input("Schedule End Date", value=date.today() + timedelta(days=2))

    submit_button = st.form_submit_button("Optimize Scheduling suggestion ⚡")


# Handle submit action
if submit_button:
    if not api_online:
        st.error("Cannot proceed: API Server is offline. Please launch the FastAPI backend first.")
    elif end_date < start_date:
        st.error("Error: Start Date must be prior to or equal to End Date.")
    else:
        # Build API payload
        payload = {
            "shoot_type": shoot_type,
            "num_people": num_people,
            "location_type": location_type,
            "lighting_setup": lighting_setup,
            "photographer_experience_years": photographer_experience_years,
            "season": season,
            "day_of_week": day_of_week,
            "lead_time_days": lead_time_days,
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": "Asia/Colombo"
        }
        
        with st.spinner("Quering API optimization engines..."):
            try:
                response = requests.post(f"{api_base_url}/schedule-suggestion", json=payload)
                if response.status_code != 200:
                    st.error(f"API Server Error ({response.status_code}): {response.text}")
                else:
                    results = response.json()
                    
                    # ---------------------------------------------------------
                    # DISPLAY RESULTS
                    # ---------------------------------------------------------
                    st.markdown("## 📊 Optimization Recommendations")
                    
                    # High-level metrics row
                    dur_pred = results["duration_prediction"]
                    price_rec = results["price_recommendation"]
                    
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.markdown(
                            f"""
                            <div class='metric-card'>
                                <div class='metric-label'>⏱️ Suggested Duration (+Buffer)</div>
                                <div class='metric-value'>{format_duration(dur_pred['predicted_duration_minutes'] + dur_pred['recommended_buffer_minutes'])}</div>
                                <div style='font-size: 0.8rem; color:#6c757d'>Base: {format_duration(dur_pred['predicted_duration_minutes'])} | Buffer: {dur_pred['recommended_buffer_minutes']} mins</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with m_col2:
                        st.markdown(
                            f"""
                            <div class='metric-card'>
                                <div class='metric-label'>💰 Optimal Quote Price</div>
                                <div class='metric-value'>${price_rec['recommended_price']:.2f}</div>
                                <div style='font-size: 0.8rem; color:#6c757d'>Expected Revenue: ${price_rec['expected_revenue']:.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with m_col3:
                        slots_count = len(results.get("best_slots", []))
                        slots_text = f"{slots_count} slots" if slots_count > 0 else "N/A (Indoor)"
                        st.markdown(
                            f"""
                            <div class='metric-card'>
                                <div class='metric-label'>📅 Optimal Daylight Slots</div>
                                <div class='metric-value'>{slots_text}</div>
                                <div style='font-size: 0.8rem; color:#6c757d'>Based on astronomical solar windows</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Details Tabs
                    tab_dur, tab_price, tab_slots = st.tabs(["⏱️ Duration SHAP Attributions", "💰 Pricing Elasticity", "📅 Ranked Slots"])
                    
                    # --- DURATION SHAP EXPLANATION ---
                    with tab_dur:
                        st.markdown("### XGBoost Duration Prediction SHAP Explanations")
                        
                        # Local SHAP calculation
                        try:
                            shap_data = explain_duration_factors(
                                shoot_type=shoot_type,
                                num_people=num_people,
                                location_type=location_type,
                                lighting_setup=lighting_setup,
                                photographer_experience_years=photographer_experience_years
                            )
                            
                            # Layout Columns
                            sd_col1, sd_col2 = st.columns([1, 1])
                            
                            with sd_col1:
                                st.write(
                                    f"The XGBoost model forecasts a **base-level** median duration of "
                                    f"**{format_duration(shap_data['base_value'])}** across all shoot formats. "
                                    f"Based on the session parameters, the following feature attributions modified this forecast to "
                                    f"arrive at **{format_duration(shap_data['prediction'])}**:"
                                )
                                
                                # Format SHAP factor descriptions
                                factors = shap_data["factors"]
                                for feat_name, val in factors.items():
                                    direction = "increases" if val >= 0 else "decreases"
                                    color = "red" if val >= 0 else "green"
                                    st.markdown(
                                        f"* **{feat_name}**: :{color}[{direction} duration by {abs(val):.1f} mins]"
                                    )
                                    
                                st.markdown(
                                    f"> **Operational Buffer Rationale:** An extra **{dur_pred['recommended_buffer_minutes']} mins** "
                                    f"buffer is recommended for **{shoot_type}** sessions. This round-up captures "
                                    f"uncertainty factors (e.g. client arrivals, transit delays) to ensure a 95% protection window."
                                )
                                
                            with sd_col2:
                                # Create Plotly horizontal bar chart for SHAP values
                                df_shap = pd.DataFrame([
                                    {"Feature": k, "SHAP Value (minutes)": v, "Color": "Positive (+)" if v >= 0 else "Negative (-)"}
                                    for k, v in factors.items()
                                ])
                                fig_shap = px.bar(
                                    df_shap,
                                    x="SHAP Value (minutes)",
                                    y="Feature",
                                    orientation="h",
                                    color="Color",
                                    color_discrete_map={"Positive (+)": "#e63946", "Negative (-)": "#2b5c3f"},
                                    title="Feature attributions (SHAP impacts)"
                                )
                                fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300)
                                st.plotly_chart(fig_shap, use_container_width=True)
                                
                        except Exception as shap_err:
                            st.warning(f"Could not compute local SHAP attributions: {shap_err}")
                            st.write(f"Predicted raw duration is {dur_pred['predicted_duration_minutes']:.1f} mins.")

                    # --- PRICING EXPLANATION ---
                    with tab_price:
                        st.markdown("### Elasticity-Based Price Adjustments")
                        
                        sp_col1, sp_col2 = st.columns([1, 1])
                        with sp_col1:
                            st.write(
                                "Dynamic pricing is calculated relative to a baseline market model. "
                                "Below are the expected price-shifter multiplier factors driving the recommendation:"
                            )
                            
                            # Reconstruct approximate multipliers based on simulator logic
                            multipliers = []
                            if season == "wedding_season":
                                multipliers.append(("Wedding Season Premium", 1.35))
                            elif season == "graduation_season":
                                multipliers.append(("Graduation Season Premium", 1.25))
                            
                            if day_of_week in ["Friday"]:
                                multipliers.append(("Weekend-Eve Premium", 1.10))
                            elif day_of_week in ["Saturday", "Sunday"]:
                                multipliers.append(("Weekend Peak Premium", 1.20))
                                
                            if lead_time_days < 7:
                                multipliers.append(("Last-Minute Convenience Premium", 1.30))
                            elif lead_time_days <= 14:
                                multipliers.append(("Short Lead-Time Premium", 1.15))
                            elif lead_time_days > 60:
                                multipliers.append(("Early-Bird Planning Discount", 0.90))
                                
                            if photographer_experience_years > 1:
                                exp_mult = 1.0 + 0.05 * photographer_experience_years
                                multipliers.append((f"Photographer Experience ({photographer_experience_years} yrs)", exp_mult))
                                
                            # Display factors list
                            if not multipliers:
                                st.markdown("* No standard price shifters applied (baseline market rate)")
                            else:
                                for label, val in multipliers:
                                    percent_diff = (val - 1.0) * 100
                                    col = "red" if percent_diff >= 0 else "green"
                                    sign = "+" if percent_diff >= 0 else ""
                                    st.markdown(f"* **{label}**: :{col}[{sign}{percent_diff:.1f}% price multiplier]")
                            
                            st.markdown(
                                f"""
                                **Demand Elasticity Analysis:**
                                * Optimal rate maximizes expected revenue (Price × Booking Acceptance Probability).
                                * Quoting a premium price increases revenue *if* accepted, but drops booking conversion rates.
                                * **Model Recommendation:** The dynamically suggested price of **${price_rec['recommended_price']:.2f}** "
                                achieves the peak statistical revenue of **${price_rec['expected_revenue']:.2f}**.
                                """
                            )
                        
                        with sp_col2:
                            # Plot expected revenue vs price curve
                            prices = np.linspace(50, 4000 if shoot_type == "wedding" else 800, 100)
                            
                            # Build rough acceptance probability curve centered around baseline
                            wtp_est = price_rec["recommended_price"] * 1.1
                            prob = 1.0 / (1.0 + np.exp(7.0 * (prices / wtp_est - 1.0)))
                            revenue = prices * prob
                            
                            fig_price = go.Figure()
                            fig_price.add_trace(go.Scatter(x=prices, y=revenue, name="Expected Revenue", line=dict(color="#2b5c3f", width=3)))
                            fig_price.add_trace(go.Scatter(x=[price_rec["recommended_price"]], y=[price_rec["expected_revenue"]],
                                                           mode="markers", name="Optimized Price Point", marker=dict(color="#e63946", size=12)))
                            fig_price.update_layout(
                                title="Expected Revenue Curve (Price vs Conversion)",
                                xaxis_title="Quoted Price ($)",
                                yaxis_title="Expected Revenue ($)",
                                height=300
                            )
                            st.plotly_chart(fig_price, use_container_width=True)

                    # --- SLOTS RANKING ---
                    with tab_slots:
                        if location_type != "outdoor":
                            st.info("Location is set to **Indoor**. Solar golden hours and meteorological forecasts are not required.")
                        else:
                            st.markdown("### Ranked Daylight Slots (7:00 AM to 6:00 PM)")
                            
                            slots_list = results.get("best_slots", [])
                            if not slots_list:
                                st.warning("No daylight slots returned for the selected dates.")
                            else:
                                # Visualization: Timeline slot quality scores
                                df_slots = pd.DataFrame(slots_list)
                                df_slots["Start Hour"] = df_slots["start_time"].apply(lambda t: t.split("T")[1][:5])
                                df_slots["Date"] = df_slots["start_time"].apply(lambda t: t.split("T")[0])
                                
                                fig_slots = px.bar(
                                    df_slots.head(15),
                                    x="score",
                                    y="start_time",
                                    orientation="h",
                                    color="score",
                                    color_continuous_scale="Viridis",
                                    title="Top 15 Time Slots (Composite Quality Score)"
                                )
                                fig_slots.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig_slots, use_container_width=True)
                                
                                # Render clean table
                                st.markdown("#### Detail Table")
                                display_rows = []
                                for idx, s in enumerate(slots_list):
                                    badge = get_weather_badge(s["score"])
                                    t_start = s["start_time"].replace("+05:30", "").replace("T", " ")
                                    t_end = s["end_time"].replace("+05:30", "").replace("T", " ")
                                    
                                    # Formatted details
                                    display_rows.append({
                                        "Rank": idx + 1,
                                        "Start Time": t_start,
                                        "End Time": t_end,
                                        "Quality Rating": badge,
                                        "Composite Score": f"{s['score']:.4f}",
                                        "Solar Score": f"{s['lighting_score']:.3f}",
                                        "Cloud Score": f"{s['cloud_score']:.3f}",
                                        "Precip Score": f"{s['precip_score']:.3f}",
                                        "Temp (°C)": f"{s['temperature_c']:.1f}°C" if not np.isnan(s['temperature_c']) else "N/A"
                                    })
                                st.dataframe(pd.DataFrame(display_rows), hide_index=True)

            except Exception as conn_err:
                st.error(f"Error querying API Gateway: {conn_err}")
