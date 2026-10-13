
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FoodWise AI",
    page_icon="🥗",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "best_model.pkl"
DATA_FILE = ROOT / "data" / "food_data.csv"
METRICS_FILE = ROOT / "models" / "metrics.json"


# ---------------- ANIMATED DESIGN ----------------
st.markdown("""
<style>
/* Animated gradient background */
.stApp {
    background: linear-gradient(
        -45deg,
        #ecfdf5,
        #fefce8,
        #dcfce7,
        #fff7ed
    );
    background-size: 300% 300%;
    animation: gradientMove 15s ease infinite;
}

@keyframes gradientMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Fade-in animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Title animation */
h1 {
    color: #166534 !important;
    font-weight: 800 !important;
    animation: fadeIn 1s ease-out;
}

/* Heading styles */
h2, h3 {
    color: #166534 !important;
    font-weight: 700 !important;
    animation: fadeIn 0.8s ease-out;
}

/* Paragraphs */
.stApp p {
    color: #374151;
}

/* Sidebar gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #dcfce7,
        #fef3c7,
        #d1fae5
    );
    background-size: 100% 200%;
    animation: sidebarGlow 10s ease infinite;
}

@keyframes sidebarGlow {
    0% { background-position: 0% 0%; }
    50% { background-position: 0% 100%; }
    100% { background-position: 0% 0%; }
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #14532d !important;
}

/* Animated buttons */
.stButton > button {
    background: linear-gradient(90deg, #16a34a, #15803d);
    background-size: 200% auto;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1rem;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(22, 163, 74, 0.18);
}

.stButton > button:hover {
    background-position: right center;
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 18px rgba(22, 163, 74, 0.3);
    color: white;
}

.stButton > button:active {
    transform: scale(0.98);
}

/* Metric card hover animation */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.94);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #bbf7d0;
    box-shadow: 0 4px 12px rgba(22, 101, 52, 0.08);
    animation: fadeIn 0.8s ease-out;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 24px rgba(22, 101, 52, 0.16);
}

[data-testid="stMetricLabel"] {
    color: #166534 !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #14532d !important;
    font-weight: 800;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    animation: fadeIn 0.8s ease-out;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
    animation: fadeIn 0.6s ease-out;
}

/* Input controls */
.stSelectbox, .stNumberInput {
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #bbf7d0;
}

/* Floating leaves decoration */
.leaf-container {
    position: fixed;
    top: 12%;
    right: 2%;
    z-index: 0;
    pointer-events: none;
    font-size: 30px;
    opacity: 0.45;
    animation: floatLeaf 5s ease-in-out infinite;
}

@keyframes floatLeaf {
    0%, 100% {
        transform: translateY(0px) rotate(-8deg);
    }
    50% {
        transform: translateY(-18px) rotate(10deg);
    }
}

/* Footer */
.footer {
    background: linear-gradient(90deg, #14532d, #15803d);
    color: white;
    text-align: center;
    padding: 18px;
    border-radius: 14px;
    margin-top: 28px;
    box-shadow: 0 5px 16px rgba(20, 83, 45, 0.2);
    animation: fadeIn 1s ease-out;
}

.footer p {
    color: white !important;
}

/* Respect reduced-motion accessibility setting */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}
</style>

<div class="leaf-container">🌿 🍃</div>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.title("🥗 FoodWise AI")
st.subheader("🌱 AI-Based Food Waste Prediction & Prevention System")

st.markdown("""
Welcome to **FoodWise AI**!

Predict meal demand, plan food preparation, and make smarter decisions
to help reduce potential food waste. 🌍
""")

st.divider()


# ---------------- LOAD MODEL ----------------
if not MODEL_FILE.exists():
    st.warning("⚠️ Machine learning model not found.")
    st.info("Run these commands from your project folder:")
    st.code(
        "python -m src.generate_data\n"
        "python -m src.train_model\n"
        "python -m streamlit run app.py",
        language="bash"
    )
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# ---------------- SIDEBAR ----------------
st.sidebar.title("🍽️ Meal Planner")
st.sidebar.markdown("### Enter Tomorrow's Details")

day = st.sidebar.selectbox(
    "📅 Day of the week",
    [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
)

menu = st.sidebar.selectbox(
    "🍛 Menu",
    ["South Indian", "North Indian", "Variety Rice", "Chapati"]
)

students = st.sidebar.number_input(
    "👩‍🎓 Expected students",
    min_value=1,
    max_value=2000,
    value=300,
    step=10
)

prepared = st.sidebar.number_input(
    "🍱 Meals you plan to prepare",
    min_value=1,
    max_value=2500,
    value=330,
    step=10
)

predict_clicked = st.sidebar.button(
    "🔍 Predict Food Demand",
    type="primary",
    use_container_width=True
)


# ---------------- PREDICTION ----------------
if predict_clicked:
    try:
        features = pd.DataFrame([{
            "day_of_week": day,
            "menu": menu,
            "students_expected": students
        }])

        predicted = max(0, round(float(model.predict(features)[0])))
        planned = int(prepared)

        surplus = max(0, planned - predicted)
        shortage = max(0, predicted - planned)
        surplus_percentage = surplus / planned * 100 if planned else 0

        st.header("📊 Prediction Results")
        st.caption("Estimated meal demand based on your selected details.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("👩‍🎓 Expected Students", f"{students:,}")

        with col2:
            st.metric("🤖 Predicted Meals Needed", f"{predicted:,}")

        with col3:
            st.metric("🍱 Planned Meals", f"{planned:,}")

        st.markdown("### 💡 Smart Recommendation")

        if surplus > 0:
            st.warning(
                f"🍱 Potential surplus: **{surplus:,} meals**. "
                "Consider adjusting preparation after checking attendance "
                "and actual demand."
            )

        elif shortage > 0:
            st.warning(
                f"⚠️ Your plan is **{shortage:,} meals below** predicted demand. "
                "Consider preparing more if the forecast is reliable."
            )

        else:
            st.success("✅ Your planned meals match the predicted demand!")

        st.progress(min(surplus_percentage / 100, 1.0))

        st.write(
            f"**Potential surplus relative to planned meals:** "
            f"{surplus_percentage:.1f}%"
        )

        st.caption(
            "Predictions are model estimates, not guaranteed outcomes. "
            "Verify them against actual canteen records."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ---------------- HISTORICAL DATA ----------------
st.divider()
st.header("📈 Historical Food Data")

if DATA_FILE.exists():
    try:
        df = pd.read_csv(DATA_FILE)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        required_columns = [
            "date", "meals_consumed", "food_waste_kg"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            st.warning(
                "Some expected dataset columns are missing: "
                + ", ".join(missing_columns)
            )
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📋 Total Records", f"{len(df):,}")

            with col2:
                avg_meals = df["meals_consumed"].mean()
                st.metric(
                    "🍽️ Average Meals Consumed",
                    f"{avg_meals:.0f}" if pd.notna(avg_meals) else "N/A"
                )

            with col3:
                avg_waste = df["food_waste_kg"].mean()
                st.metric(
                    "♻️ Average Recorded Waste",
                    f"{avg_waste:.1f} kg"
                    if pd.notna(avg_waste) else "N/A"
                )

            st.subheader("🍽️ Meals Consumed Over Time")

            chart_df = (
                df.dropna(subset=["date"])
                .sort_values("date")
                .set_index("date")[["meals_consumed"]]
            )

            if not chart_df.empty:
                st.line_chart(chart_df)
            else:
                st.info("No valid dates are available for the chart.")

            st.subheader("🗂️ Recent Dataset Records")

            st.dataframe(
                df.tail(10),
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:
        st.error(f"Could not read historical data: {error}")

else:
    st.info("Historical dataset not found. Generate the dataset to view charts.")


# ---------------- MODEL EVALUATION ----------------
st.divider()
st.header("🤖 Model Evaluation")

if METRICS_FILE.exists():
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        st.markdown("### 🧠 Best Performing Model")
        st.success(
            f"Selected model: {metrics.get('best_model', 'Not available')}"
        )

        if "test_rows" in metrics:
            st.caption(
                f"Evaluation used {metrics['test_rows']} held-out sample records."
            )

        results = metrics.get("results", [])

        if results:
            st.subheader("📊 Model Performance Comparison")
            st.dataframe(
                pd.DataFrame(results),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No model comparison results are available.")

        st.caption(
            "MAE and RMSE measure prediction errors in meals. "
            "Lower values are generally better. R² can be negative "
            "when predictions perform poorly."
        )

    except Exception as error:
        st.error(f"Could not read model evaluation results: {error}")

else:
    st.info("Model evaluation metrics not found. Train the models first.")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <p>🥗 <b>FoodWise AI</b></p>
    <p>Smarter Meal Planning • Less Food Waste • A Greener Future 🌱</p>
</div>
""", unsafe_allow_html=True)
