from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "food_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Dataset missing. Run: python -m src.generate_data")
    df = pd.read_csv(DATA_FILE)
    features = ["day_of_week", "menu", "students_expected"]
    target = "meals_consumed"
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    categorical, numeric = ["day_of_week", "menu"], ["students_expected"]
    estimators = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=3),
    }
    results, best_mae, best_name, best_pipeline = [], float("inf"), None, None
    for name, estimator in estimators.items():
        preprocessor = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ])
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)
        results.append({"model": name, "MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2), "R2": round(float(r2), 3)})
        print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
        if mae < best_mae:
            best_mae, best_name, best_pipeline = mae, name, pipeline
    joblib.dump(best_pipeline, MODEL_DIR / "best_model.pkl")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({"best_model": best_name, "results": results, "test_rows": len(y_test)}, file, indent=2)
    print(f"\nBest model: {best_name}\nSaved model to: {MODEL_DIR / 'best_model.pkl'}")

if __name__ == "__main__":
    main()
