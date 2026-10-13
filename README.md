# FoodWise AI — Food Waste Prediction System

A beginner-friendly educational prototype using Python, Pandas, scikit-learn, and Streamlit. It creates synthetic canteen data, compares Linear Regression and Random Forest models, and displays estimated meal demand in a dashboard.

## Requirements
- Python 3.10–3.12 recommended
- Windows, macOS, or Linux

## Run on Windows (Command Prompt)
From this project folder:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
py -m pip install -r requirements.txt
py -m src.generate_data
py -m src.train_model
py -m streamlit run app.py
```

## Important limitation
The included dataset is synthetic demonstration data, not real canteen data. Model metrics and predictions are only for learning and must not be represented as validated real-world performance. Replace it with suitable historical canteen data and evaluate before making operational decisions.
