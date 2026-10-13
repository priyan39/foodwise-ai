from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def generate_data():
    rng = np.random.default_rng(42)
    menus = ["South Indian", "North Indian", "Variety Rice", "Chapati"]
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    rows = []
    for date in dates:
        day = date.day_name()
        menu = rng.choice(menus)
        base_students = rng.integers(220, 351)
        if day == "Sunday":
            base_students = rng.integers(80, 151)
        elif day == "Saturday":
            base_students = int(base_students * 0.65)
        students_expected = int(base_students)
        meals_consumed = int(np.clip(
            students_expected * rng.uniform(0.84, 0.99) + rng.normal(0, 8),
            30, students_expected
        ))
        meals_prepared = int(meals_consumed + rng.integers(5, 46))
        food_waste_kg = round(max(0, (meals_prepared - meals_consumed) * rng.uniform(0.12, 0.22)), 2)
        rows.append({
            "date": date.strftime("%Y-%m-%d"), "day_of_week": day,
            "menu": menu, "students_expected": students_expected,
            "meals_prepared": meals_prepared, "meals_consumed": meals_consumed,
            "food_waste_kg": food_waste_kg,
        })
    df = pd.DataFrame(rows)
    output = DATA_DIR / "food_data.csv"
    df.to_csv(output, index=False)
    print(f"Dataset created: {output}\nTotal records: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    generate_data()
