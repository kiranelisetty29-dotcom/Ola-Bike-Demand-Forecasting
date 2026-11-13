import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path
from datetime import date

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Ola Bike Demand Forecasting",
    page_icon="🏍️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "hour_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "ola_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "features.pkl"


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3F6FC;
    }

    [data-testid="stSidebar"] {
        background-color: #14213D;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #14213D;
    }

    .subtitle {
        color: #64748B;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }

    .stButton button {
        background-color: #1769E0;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 3. LOAD DATA AND MODEL
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features


try:
    df = load_data()
    model, features = load_model()

except Exception as error:
    st.error("Dataset or model could not be loaded.")
    st.exception(error)
    st.stop()


# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:
    st.title("🏍️ OLA FORECAST")
    st.caption("Bike Demand Prediction System")
    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
        [
            "Home Dashboard",
            "Ride Prediction",
            "Demand Analysis",
            "About Project"
        ]
    )

    st.markdown("---")
    st.success("ML Model Connected")


# ==========================================
# 5. HOME DASHBOARD
# ==========================================

if page == "Home Dashboard":

    st.markdown(
        '<div class="main-title">Home Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Bike demand overview and historical analytics'
        '</div>',
        unsafe_allow_html=True
    )

    total_rides = int(df["cnt"].sum())
    average_rides = round(df["cnt"].mean(), 2)
    peak_hour = int(df.groupby("hr")["cnt"].mean().idxmax())
    max_demand = int(df["cnt"].max())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Bike Rentals", f"{total_rides:,}")
    col2.metric("Average Demand", average_rides)
    col3.metric("Peak Demand Hour", f"{peak_hour}:00")
    col4.metric("Maximum Hourly Demand", max_demand)

    st.markdown("---")

    hourly = (
        df.groupby("hr")["cnt"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        hourly,
        x="hr",
        y="cnt",
        markers=True,
        title="Average Bike Demand by Hour",
        labels={
            "hr": "Hour of Day",
            "cnt": "Average Rentals"
        }
    )

    fig.update_traces(
        line_color="#1769E0",
        line_width=3
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Demand Overview")

    col1, col2 = st.columns(2)
    col1.info(f"Peak demand hour: {peak_hour}:00")
    col2.info(f"Average demand: {average_rides}")


# ==========================================
# 6. RIDE PREDICTION
# ==========================================

elif page == "Ride Prediction":

    st.markdown(
        '<div class="main-title">Ride Demand Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Select any date between 1900 and 2100 "
        "to estimate hourly bike demand."
    )

    st.warning(
        "This model uses historical 2011–2012 bike-sharing "
        "patterns. Selecting a future year does not mean "
        "the model has learned actual future demand trends."
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input(
            "Select Prediction Date",
            value=date.today(),
            min_value=date(1900, 1, 1),
            max_value=date(2100, 12, 31),
            format="DD/MM/YYYY"
        )

        hour = st.slider(
            "Select Hour",
            min_value=0,
            max_value=23,
            value=9
        )

        temperature = st.slider(
            "Temperature (°C)",
            min_value=0.0,
            max_value=41.0,
            value=25.0
        )

        humidity = st.slider(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=50
        )

    with col2:
        weather_options = {
            "Clear / Partly Cloudy": 1,
            "Mist / Cloudy": 2,
            "Light Rain / Snow": 3,
            "Heavy Rain / Snow": 4
        }

        weather = st.selectbox(
            "Weather Condition",
            list(weather_options.keys())
        )

        windspeed = st.slider(
            "Wind Speed (km/h)",
            min_value=0.0,
            max_value=67.0,
            value=15.0
        )

        holiday = st.selectbox(
            "Is it a Holiday?",
            ["No", "Yes"]
        )

    st.markdown("---")

    if st.button("Predict Bike Demand", use_container_width=True):

        month = selected_date.month

        weekday = (selected_date.weekday() + 1) % 7

        # UCI season mapping
        if month in [3, 4, 5]:
            season = 1
        elif month in [6, 7, 8]:
            season = 2
        elif month in [9, 10, 11]:
            season = 3
        else:
            season = 4

        is_holiday = 1 if holiday == "Yes" else 0

        workingday = (
            0 if weekday in [0, 6] or is_holiday else 1
        )

        # Historical model encoding:
        # 0 = 2011 pattern, 1 = 2012 pattern
        yr = 0 if selected_date.year <= 2011 else 1

        input_values = {
            "season": season,
            "yr": yr,
            "mnth": month,
            "hr": hour,
            "holiday": is_holiday,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weather_options[weather],
            "temp": temperature / 41.0,
            "hum": humidity / 100.0,
            "windspeed": windspeed / 67.0
        }

        try:
            input_data = pd.DataFrame([input_values])

            # Match the feature order used during model training.
            input_data = input_data[features]

            prediction = float(model.predict(input_data)[0])
            prediction = max(0, prediction)
            predicted_rides = round(prediction)

            low = df["cnt"].quantile(0.33)
            high = df["cnt"].quantile(0.67)

            st.markdown("---")
            st.subheader("Prediction Result")

            st.metric(
                "Expected Hourly Bike Rentals",
                f"{predicted_rides:,}"
            )

            if prediction <= low:
                st.success("LOW DEMAND")
            elif prediction <= high:
                st.warning("MODERATE DEMAND")
            else:
                st.error("HIGH DEMAND")

            st.write(
                f"Selected Date: {selected_date.strftime('%d-%m-%Y')}"
            )
            st.write(f"Selected Hour: {hour}:00")
            st.write(f"Weather: {weather}")

            st.caption(
                "Prediction is an estimate based on historical "
                "bike-sharing patterns, not guaranteed Ola requests."
            )

        except Exception as error:
            st.error("Prediction failed. Please check model features.")
            st.exception(error)


# ==========================================
# 7. DEMAND ANALYSIS
# ==========================================

elif page == "Demand Analysis":

    st.markdown(
        '<div class="main-title">Demand Analysis</div>',
        unsafe_allow_html=True
    )

    st.write("Explore historical bike demand patterns.")

    # Hourly demand
    hourly = (
        df.groupby("hr")["cnt"]
        .mean()
        .reset_index()
    )

    fig1 = px.bar(
        hourly,
        x="hr",
        y="cnt",
        title="Hourly Bike Demand",
        labels={
            "hr": "Hour",
            "cnt": "Average Rentals"
        },
        color="cnt",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Weekday demand
    weekday_data = (
        df.groupby("weekday")["cnt"]
        .mean()
        .reset_index()
    )

    names = {
        0: "Sunday",
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday"
    }

    weekday_data["Day"] = weekday_data["weekday"].map(names)

    fig2 = px.bar(
        weekday_data,
        x="Day",
        y="cnt",
        title="Bike Demand by Weekday"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Weather demand
    weather_data = (
        df.groupby("weathersit")["cnt"]
        .mean()
        .reset_index()
    )

    weather_names = {
        1: "Clear",
        2: "Mist",
        3: "Light Rain/Snow",
        4: "Heavy Rain/Snow"
    }

    weather_data["Weather"] = (
        weather_data["weathersit"].map(weather_names)
    )

    fig3 = px.bar(
        weather_data,
        x="Weather",
        y="cnt",
        title="Weather vs Bike Demand"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Monthly demand
    monthly = (
        df.groupby("mnth")["cnt"]
        .mean()
        .reset_index()
    )

    fig4 = px.line(
        monthly,
        x="mnth",
        y="cnt",
        markers=True,
        title="Monthly Bike Demand"
    )

    st.plotly_chart(fig4, use_container_width=True)


# ==========================================
# 8. ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.markdown(
        '<div class="main-title">About Project</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Ola Bike Ride Request Forecasting is a "
        "Machine Learning based prototype designed "
        "to estimate hourly bike-sharing demand."
    )

    st.subheader("Project Objectives")

    st.markdown(
        """
        - Predict hourly bike demand.
        - Identify peak-demand periods.
        - Analyze weather effects.
        - Compare weekday and weekend patterns.
        - Support bike availability planning.
        """
    )

    st.subheader("Technologies Used")

    st.markdown(
        """
        - Python
        - Streamlit
        - Pandas and NumPy
        - Scikit-learn
        - Plotly
        - Joblib
        """
    )

    st.subheader("Machine Learning Algorithms")

    st.markdown(
        """
        1. Linear Regression
        2. Decision Tree Regression
        3. Random Forest Regression
        """
    )

    st.subheader("Project Workflow")

    st.write(
        "Dataset → Data Cleaning → Data Analysis → "
        "Model Training → Model Evaluation → "
        "Demand Prediction"
    )

    st.warning(
        "This prototype uses the public UCI Bike Sharing Dataset, "
        "not actual private Ola booking records."
    )


# ==========================================
# 9. FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    "<center>Ola Bike Ride Demand Forecasting | "
    "Machine Learning Project</center>",
    unsafe_allow_html=True
)