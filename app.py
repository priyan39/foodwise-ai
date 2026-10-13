from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "best_model.pkl"
DATA_FILE = ROOT / "data" / "food_data.csv"
METRICS_FILE = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="FoodWise AI", page_icon="🥗", layout="wide")
st.title("🥗 FoodWise AI")
st.subheader("AI-Based Food Waste Prediction & Prevention System")
st.write("Predict meal demand and plan food preparation to help reduce potential food waste.")

if not MODEL_FILE.exists():
    st.warning("ML model not found. Run these commands in the project folder:")
    st.code("python -m src.generate_data\npython -m src.train_model\npython -m streamlit run app.py")
    st.stop()

model = joblib.load(MODEL_FILE)
st.sidebar.header("🍽️ Tomorrow's Details")
day = st.sidebar.selectbox("Day of the week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
menu = st.sidebar.selectbox("Menu", ["South Indian", "North Indian", "Variety Rice", "Chapati"])
students = st.sidebar.number_input("Expected students", min_value=1, max_value=2000, value=300, step=10)
prepared = st.sidebar.number_input("Meals you plan to prepare", min_value=1, max_value=2500, value=330, step=10)

if st.sidebar.button("Predict Food Demand", type="primary"):
    features = pd.DataFrame([{"day_of_week": day, "menu": menu, "students_expected": students}])
    predicted = max(0, round(float(model.predict(features)[0])))
    planned = int(prepared)
    surplus = max(0, planned - predicted)
    shortage = max(0, predicted - planned)
    st.header("📊 Prediction Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Students", f"{students:,}")
    col2.metric("Predicted Meals Needed", f"{predicted:,}")
    col3.metric("Planned Meals", f"{planned:,}")
    if surplus > 0:
        st.warning(f"Potential surplus: {surplus} meals. Consider adjusting preparation after checking attendance and demand.")
    elif shortage > 0:
        st.warning(f"Your plan is {shortage} meals below predicted demand. Consider preparing more if the forecast is reliable.")
    else:
        st.success("Planned meals match the predicted demand.")
    st.write(f"Potential surplus relative to planned meals: {surplus / planned * 100:.1f}%")
    st.caption("These are model estimates, not guaranteed outcomes. Verify predictions against actual canteen records.")

st.divider()
st.header("📈 Historical Data")
if DATA_FILE.exists():
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Average meals consumed", f"{df['meals_consumed'].mean():.0f}")
    col3.metric("Average recorded waste", f"{df['food_waste_kg'].mean():.1f} kg")
    st.subheader("Meals Consumed Over Time")
    st.line_chart(df.set_index("date")[["meals_consumed"]])
    st.subheader("Sample Dataset")
    st.dataframe(df.tail(10), use_container_width=True)
else:
    st.info("Generate the dataset to view historical charts.")

st.divider()
st.header("🤖 Model Evaluation")
if METRICS_FILE.exists():
    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    st.write(f"**Selected model:** {metrics['best_model']}")
    st.caption(f"Evaluation used {metrics['test_rows']} held-out sample records.")
    st.dataframe(pd.DataFrame(metrics["results"]), use_container_width=True, hide_index=True)
    st.caption("MAE and RMSE measure prediction errors in meals. Lower is better. R² can be negative on poor predictions.")
else:
    st.info("Train the models to see evaluation metrics.")
st.caption("FoodWise AI | Educational prototype")
