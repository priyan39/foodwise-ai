
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# ==================================================
# 1. PROJECT PATHS
# ==================================================
ROOT = Path(__file__).resolve().parent

MODEL_FILE = ROOT / "models" / "best_model.pkl"
DATA_FILE = ROOT / "data" / "food_data.csv"
METRICS_FILE = ROOT / "models" / "metrics.json"
ACTUAL_FILE = ROOT / "data" / "actual_meals.csv"
DONATION_FILE = ROOT / "data" / "donation_listings.csv"

# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="FoodWise AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 3. ANIMATED GREEN THEME
# ==================================================
st.markdown("""
<style>
/* Animated page background */
.stApp {
    background: linear-gradient(
        135deg,
        #f0fdf4,
        #ecfeff,
        #f5f3ff,
        #f0fdf4
    );
    background-size: 300% 300%;
    animation: gradientMove 14s ease infinite;
    color: #1f2937;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Main title */
.stApp h1 {
    color: #14532d !important;
    font-weight: 800 !important;
}

.stApp h2,
.stApp h3 {
    color: #166534 !important;
    font-weight: 700 !important;
}

/* Normal text */
.stApp p,
.stApp li {
    color: #1f2937;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #dcfce7;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #14532d !important;
}

/* Sidebar navigation selection */
section[data-testid="stSidebar"] [aria-checked="true"] p {
    color: #166534 !important;
    font-weight: 700;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.96);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #bbf7d0;
    box-shadow: 0 5px 18px rgba(22, 101, 52, 0.10);
}

div[data-testid="stMetric"] label {
    color: #374151 !important;
}

div[data-testid="stMetricValue"] {
    color: #166534 !important;
    font-weight: 800 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #15803d, #059669);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    transition: transform 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #166534;
    color: white !important;
    transform: translateY(-2px);
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: white;
    color: #111827;
    border-radius: 8px;
}

/* Data tables */
div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 12px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Progress bar */
div[data-testid="stProgressBar"] > div > div {
    background-color: #16a34a;
}

/* Dividers */
hr {
    border-color: #bbf7d0;
}

/* Footer */
.footer {
    text-align: center;
    color: #166534;
    padding: 18px;
    font-size: 14px;
}

/* Reduce animation for accessibility settings */
@media (prefers-reduced-motion: reduce) {
    .stApp {
        animation: none;
    }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 4. LOAD MODEL AND DATA
# ==================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    if "date" in data.columns:
        data["date"] = pd.to_datetime(
            data["date"], errors="coerce"
        )

    return data

model = None
df = pd.DataFrame()

if MODEL_FILE.exists():
    try:
        model = load_model(str(MODEL_FILE))
    except Exception as error:
        st.error(f"Model loading failed: {error}")

if DATA_FILE.exists():
    try:
        df = load_data(str(DATA_FILE))
    except Exception as error:
        st.error(f"Dataset loading failed: {error}")

# ==================================================
# 5. HEADER
# ==================================================
st.title("🥗 FoodWise AI")
st.subheader("AI-Based Food Waste Prediction & Prevention System")

st.write(
    "Predict meal demand, plan food preparation, track food waste, "
    "and discover ways to reduce surplus food."
)

st.info(
    "🌱 An educational AI prototype for smarter food planning "
    "and sustainable food management."
)

# ==================================================
# 6. SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("🌱 FoodWise AI")
st.sidebar.caption("Smart food. Less waste. Better future.")

page = st.sidebar.radio(
    "Choose a feature",
    [
        "Dashboard",
        "Demand Predictor",
        "Food Waste Analytics",
        "Menu Planner",
        "Food Donation",
        "Environmental Impact",
        "FoodWise Assistant",
        "Eco Challenge",
        "Model Evaluation"
    ]
)

st.sidebar.divider()
st.sidebar.caption("♻️ Small actions make a big difference!")

# ==================================================
# 7. HELPER: PREDICT DEMAND
# ==================================================
def predict_demand(day, menu, students):
    if model is None:
        raise RuntimeError(
            "Trained model is not available. Check models/best_model.pkl."
        )

    features = pd.DataFrame([{
        "day_of_week": day,
        "menu": menu,
        "students_expected": students
    }])

    result = float(model.predict(features)[0])
    return max(0, round(result))

# ==================================================
# 8. DASHBOARD
# ==================================================
if page == "Dashboard":

    st.header("📊 Project Dashboard")

    if not df.empty:
        col1, col2, col3 = st.columns(3)

        col1.metric("📁 Dataset Records", f"{len(df):,}")

        if "meals_consumed" in df.columns:
            average = pd.to_numeric(
                df["meals_consumed"], errors="coerce"
            ).mean()

            col2.metric(
                "🍚 Average Meals Consumed",
                f"{average:,.0f}" if pd.notna(average) else "N/A"
            )
        else:
            col2.metric("🍚 Average Meals Consumed", "N/A")

        if "food_waste_kg" in df.columns:
            average_waste = pd.to_numeric(
                df["food_waste_kg"], errors="coerce"
            ).mean()

            col3.metric(
                "♻️ Average Food Waste",
                f"{average_waste:.1f} kg"
                if pd.notna(average_waste) else "N/A"
            )
        else:
            col3.metric("♻️ Average Food Waste", "N/A")

        if "date" in df.columns and "meals_consumed" in df.columns:
            st.subheader("📈 Meals Consumed Over Time")

            chart_data = df.dropna(subset=["date"]).sort_values("date")

            if not chart_data.empty:
                st.line_chart(
                    chart_data.set_index("date")[["meals_consumed"]]
                )

        if "food_waste_kg" in df.columns:
            st.subheader("🗑️ Food Waste Overview")
            waste = pd.to_numeric(
                df["food_waste_kg"], errors="coerce"
            ).dropna()

            if not waste.empty:
                st.bar_chart(waste.tail(20).reset_index(drop=True))

        st.subheader("📋 Recent Dataset Records")
        st.dataframe(df.tail(10), use_container_width=True)

    else:
        st.warning(
            "Dataset not found. Check the data/food_data.csv file."
        )

    st.divider()
    st.subheader("✨ What FoodWise AI Can Do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔮 Demand Prediction")
        st.write("Estimate how many meals may be required.")

    with col2:
        st.markdown("### ♻️ Waste Prevention")
        st.write("Review food waste and improve meal planning.")

    with col3:
        st.markdown("### 🤝 Food Donation")
        st.write("Register surplus food for donation follow-up.")

# ==================================================
# 9. DEMAND PREDICTOR
# ==================================================
elif page == "Demand Predictor":

    st.header("🔮 AI Food Demand Predictor")
    st.write("Enter tomorrow's details to estimate meal demand.")

    with st.form("demand_form"):

        day = st.selectbox(
            "📅 Day of the week",
            [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ]
        )

        menu = st.selectbox(
            "🍱 Menu",
            [
                "South Indian",
                "North Indian",
                "Variety Rice",
                "Chapati"
            ]
        )

        students = st.number_input(
            "👨‍🎓 Expected students",
            min_value=1,
            max_value=2000,
            value=300,
            step=10
        )

        prepared = st.number_input(
            "🍚 Meals planned for preparation",
            min_value=1,
            max_value=2500,
            value=330,
            step=10
        )

        submitted = st.form_submit_button(
            "🚀 Predict Food Demand",
            type="primary"
        )

    if submitted:
        try:
            predicted = predict_demand(day, menu, students)

            surplus = max(0, int(prepared) - predicted)
            shortage = max(0, predicted - int(prepared))

            st.divider()
            st.subheader("📊 Prediction Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Expected Students", f"{students:,}")
            col2.metric("Predicted Meals Needed", f"{predicted:,}")
            col3.metric("Planned Meals", f"{prepared:,}")

            if surplus > 0:
                st.warning(
                    f"⚠️ Potential surplus: {surplus} meals. "
                    "Review the preparation quantity."
                )
            elif shortage > 0:
                st.warning(
                    f"⚠️ Potential shortage: {shortage} meals. "
                    "Review the planned quantity."
                )
            else:
                st.success(
                    "✅ Planned meals match the predicted demand!"
                )

            st.metric(
                "Potential Surplus vs Planned Meals",
                f"{surplus / prepared * 100:.1f}%"
            )

            st.caption(
                "Predictions are estimates. Compare them with actual "
                "attendance and canteen records before making decisions."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# ==================================================
# 10. FOOD WASTE ANALYTICS
# ==================================================
elif page == "Food Waste Analytics":

    st.header("♻️ Food Waste Analytics")

    if not df.empty and "food_waste_kg" in df.columns:

        waste = pd.to_numeric(
            df["food_waste_kg"], errors="coerce"
        ).dropna()

        if not waste.empty:
            col1, col2, col3 = st.columns(3)

            col1.metric("Total Recorded Waste", f"{waste.sum():.1f} kg")
            col2.metric("Average Waste", f"{waste.mean():.1f} kg")
            col3.metric("Highest Record", f"{waste.max():.1f} kg")

            st.subheader("Waste Records")
            st.line_chart(waste.reset_index(drop=True))

        else:
            st.info("No valid waste records found.")

    else:
        st.info(
            "The historical dataset needs a food_waste_kg column "
            "to display waste analytics."
        )

    st.divider()
    st.subheader("📝 Record Actual Meal Results")

    with st.form("actual_form"):

        actual_date = st.date_input("Date")

        actual_prepared = st.number_input(
            "Meals prepared", min_value=0, value=300
        )

        actual_consumed = st.number_input(
            "Meals consumed", min_value=0, value=270
        )

        actual_waste = st.number_input(
            "Measured food waste (kg)",
            min_value=0.0,
            value=5.0,
            step=0.5
        )

        save_actual = st.form_submit_button("💾 Save Actual Results")

    if save_actual:

        if actual_consumed > actual_prepared:
            st.error("Meals consumed cannot exceed meals prepared.")

        else:
            ACTUAL_FILE.parent.mkdir(parents=True, exist_ok=True)

            record = pd.DataFrame([{
                "date": str(actual_date),
                "meals_prepared": actual_prepared,
                "meals_consumed": actual_consumed,
                "food_waste_kg": actual_waste
            }])

            try:
                if ACTUAL_FILE.exists():
                    record.to_csv(
                        ACTUAL_FILE,
                        mode="a",
                        header=False,
                        index=False
                    )
                else:
                    record.to_csv(ACTUAL_FILE, index=False)

                st.success("✅ Actual meal results saved!")

            except Exception as error:
                st.error(f"Could not save results: {error}")

    if ACTUAL_FILE.exists():
        st.subheader("📋 Saved Actual Results")

        try:
            actual_df = pd.read_csv(ACTUAL_FILE)
            st.dataframe(actual_df.tail(20), use_container_width=True)
        except Exception as error:
            st.warning(f"Could not display saved results: {error}")

    st.caption(
        "Cloud-hosted local files may not persist across restarts "
        "or redeployments."
    )

# ==================================================
# 11. MENU PLANNER
# ==================================================
elif page == "Menu Planner":

    st.header("🍱 Smart Meal Preparation Planner")

    st.write(
        "Compare the model's estimated demand with your planned meals."
    )

    students = st.number_input(
        "Expected students",
        min_value=1,
        max_value=2000,
        value=300,
        step=10,
        key="planner_students"
    )

    day = st.selectbox(
        "Day",
        [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ],
        key="planner_day"
    )

    menu = st.selectbox(
        "Menu",
        [
            "South Indian",
            "North Indian",
            "Variety Rice",
            "Chapati"
        ],
        key="planner_menu"
    )

    prepared = st.number_input(
        "Planned meals",
        min_value=1,
        max_value=2500,
        value=330,
        step=10,
        key="planner_prepared"
    )

    if st.button("📋 Generate Meal Plan"):

        try:
            predicted = predict_demand(day, menu, students)
            difference = int(prepared) - predicted

            col1, col2 = st.columns(2)
            col1.metric("Estimated Demand", f"{predicted:,}")
            col2.metric("Planned Meals", f"{prepared:,}")

            if difference > 0:
                st.warning(
                    f"Review a potential surplus of {difference} meals."
                )
            elif difference < 0:
                st.warning(
                    f"Review a potential shortage of {abs(difference)} meals."
                )
            else:
                st.success("Your planned meals match estimated demand.")

        except Exception as error:
            st.error(f"Could not create meal plan: {error}")

    st.info(
        "Use attendance records and practical experience alongside "
        "the model prediction."
    )

# ==================================================
# 12. FOOD DONATION REGISTER
# ==================================================
elif page == "Food Donation":

    st.header("🤝 Surplus Food Donation Register")

    st.write(
        "Register surplus food so your team can follow up with "
        "a verified local food rescue organisation."
    )

    with st.form("donation_form"):

        donor = st.text_input("Canteen / Donor name")
        food = st.text_input("Food available")

        quantity = st.number_input(
            "Quantity (kg)",
            min_value=0.1,
            value=2.0,
            step=0.5
        )

        area = st.text_input("Pickup area")
        contact = st.text_input("Contact details")

        preparation_time = st.selectbox(
            "Preparation time",
            [
                "Prepared recently",
                "Prepared earlier",
                "Not specified"
            ]
        )

        submitted = st.form_submit_button("🤝 Register Surplus Food")

    if submitted:

        if not donor.strip() or not food.strip() or not area.strip():
            st.error("Enter the donor name, food name, and pickup area.")

        else:
            DONATION_FILE.parent.mkdir(parents=True, exist_ok=True)

            listing = pd.DataFrame([{
                "donor": donor.strip(),
                "food": food.strip(),
                "quantity_kg": quantity,
                "pickup_area": area.strip(),
                "contact": contact.strip(),
                "preparation_time": preparation_time,
                "status": "Pending contact"
            }])

            try:
                if DONATION_FILE.exists():
                    listing.to_csv(
                        DONATION_FILE,
                        mode="a",
                        header=False,
                        index=False
                    )
                else:
                    listing.to_csv(DONATION_FILE, index=False)

                st.success("✅ Donation listing registered!")
                st.warning(
                    "This form does not contact NGOs automatically. "
                    "Confirm recipient availability, food safety, "
                    "storage, and pickup before donating."
                )

            except Exception as error:
                st.error(f"Could not register donation: {error}")

    if DONATION_FILE.exists():

        st.subheader("📋 Registered Food Listings")

        try:
            listings = pd.read_csv(DONATION_FILE)
            st.dataframe(listings, use_container_width=True)
        except Exception as error:
            st.warning(f"Could not display listings: {error}")

# ==================================================
# 13. ENVIRONMENTAL IMPACT CALCULATOR
# ==================================================
elif page == "Environmental Impact":

    st.header("🌍 Environmental Impact Calculator")

    st.write(
        "Estimate the possible environmental impact of avoiding food waste."
    )

    saved = st.number_input(
        "Food waste avoided (kg)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    # Illustrative assumption, not a measured emissions factor.
    factor = 2.5
    estimated_co2 = saved * factor

    col1, col2 = st.columns(2)

    col1.metric("Food Waste Avoided", f"{saved:.1f} kg")
    col2.metric("Illustrative CO₂e Avoided", f"{estimated_co2:.1f} kg")

    st.caption(
        "Illustrative estimate using an assumed factor of 2.5 kg CO₂e "
        "per kg of food waste avoided. Actual impact varies by food type, "
        "production, transport, and disposal method."
    )

    st.success(
        "🌱 Preventing avoidable food waste can also save ingredients, "
        "water, energy, and money."
    )

# ==================================================
# 14. FOODWISE ASSISTANT
# ==================================================
elif page == "FoodWise Assistant":

    st.header("💬 FoodWise Assistant")

    st.write(
        "Get simple suggestions about food waste, demand prediction, "
        "donation, and sustainability."
    )

    st.caption(
        "This is a rule-based demo assistant, not a live generative AI model."
    )

    question = st.text_input(
        "Ask your question",
        placeholder="Example: How can I reduce food waste?"
    )

    if st.button("✨ Get Suggestion"):

        q = question.lower().strip()

        if not q:
            st.warning("Please enter a question.")

        elif any(word in q for word in [
            "waste", "leftover", "surplus"
        ]):
            st.markdown("### ♻️ Food Waste Reduction Tips")
            st.write(
                "1. Estimate attendance before cooking.\n\n"
                "2. Prepare food in suitable batches.\n\n"
                "3. Measure leftover food daily.\n\n"
                "4. Review menus that frequently create leftovers.\n\n"
                "5. Store food safely and follow food safety guidance."
            )

        elif any(word in q for word in [
            "predict", "demand", "quantity", "meal"
        ]):
            st.markdown("### 🔮 Meal Demand Tips")
            st.write(
                "Open Demand Predictor, select the day and menu, "
                "enter expected attendance, and compare the prediction "
                "with actual canteen records."
            )

        elif any(word in q for word in [
            "donate", "donation", "ngo"
        ]):
            st.markdown("### 🤝 Donation Tips")
            st.write(
                "Contact a verified local food rescue organisation. "
                "Confirm collection time, recipient availability, "
                "food safety, and storage requirements before donating."
            )

        elif any(word in q for word in [
            "environment", "climate", "carbon"
        ]):
            st.markdown("### 🌍 Sustainability Tips")
            st.write(
                "Reducing avoidable food waste can reduce the resources "
                "used to produce and transport food. The impact calculator "
                "provides an illustrative estimate."
            )

        else:
            st.write(
                "Try asking about food waste, meal demand, donation, "
                "or environmental impact."
            )

# ==================================================
# 15. ECO CHALLENGE
# ==================================================
elif page == "Eco Challenge":

    st.header("🏆 7-Day Food-Saving Challenge")

    st.write(
        "Complete these actions with your team to build better food habits."
    )

    tasks = [
        "Day 1: Record the amount of leftover food.",
        "Day 2: Estimate attendance before meal preparation.",
        "Day 3: Identify frequently wasted menu items.",
        "Day 4: Review whether smaller preparation batches are practical.",
        "Day 5: Check food storage and safety practices.",
        "Day 6: Contact a verified food rescue organisation.",
        "Day 7: Compare food waste records across the week."
    ]

    completed = 0

    for index, task in enumerate(tasks):
        if st.checkbox(task, key=f"eco_task_{index}"):
            completed += 1

    st.progress(completed / len(tasks))
    st.metric("Challenge Progress", f"{completed}/{len(tasks)} days")

    if completed == len(tasks):
        st.success("🎉 Challenge completed! Keep tracking your results.")

    st.caption(
        "Progress is kept for the active app session and is not "
        "saved as a permanent user profile."
    )

# ==================================================
# 16. MODEL EVALUATION
# ==================================================
elif page == "Model Evaluation":

    st.header("🧠 Machine Learning Model Evaluation")

    if METRICS_FILE.exists():

        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as file:
                metrics = json.load(file)

            st.metric(
                "Selected Model",
                str(metrics.get("best_model", "Not specified"))
            )

            if "test_rows" in metrics:
                st.metric(
                    "Held-out Test Records",
                    str(metrics["test_rows"])
                )

            results = metrics.get("results", [])

            if isinstance(results, list) and results:

                results_df = pd.DataFrame(results)

                st.subheader("📊 Model Comparison")
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "MAE and RMSE measure prediction error; lower values "
                    "are generally better. R² describes how well predictions "
                    "fit the evaluation data. Interpret all metrics together."
                )

            else:
                st.info(
                    "No model comparison results found in metrics.json."
                )

        except Exception as error:
            st.error(f"Could not read model metrics: {error}")

    else:
        st.info(
            "Model evaluation metrics are missing. Run "
            "`python -m src.train_model` from your project folder."
        )

# ==================================================
# 17. FOOTER
# ==================================================
st.divider()

st.markdown("""
<div class="footer">
    🥗 <b>FoodWise AI</b><br>
    AI + Sustainability + Food Waste Prevention<br>
    <small>Student Project Prototype</small>
</div>
""", unsafe_allow_html=True)
