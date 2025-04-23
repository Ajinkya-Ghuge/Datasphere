import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import datetime

# Synthetic training data
def generate_dummy_data():
    data = {
        'days_to_expiry': [30, 20, 15, 10, 5, 25, 18, 12],
        'temp_diff': [0, 5, 10, 15, 20, 10, 5, 15],
        'days_since_purchase': [5, 3, 2, 1, 0, 4, 3, 2]
    }
    return pd.DataFrame(data)

# Train Random Forest model (no split for now)
def train_model():
    df = generate_dummy_data()
    X = df[['temp_diff', 'days_since_purchase']]
    y = df['days_to_expiry']

    # Train on all data (8 rows) since it's small
    model = RandomForestRegressor(n_estimators=50, random_state=42)  # More trees
    model.fit(X, y)

    # Simulate a "test" by predicting on training data (for demo)
    y_pred = model.predict(X)
    mae = sum(abs(y_pred[i] - y[i]) for i in range(len(y))) / len(y)
    print(f"Training MAE (self-test): {mae:.2f} days")
    print(f"Predictions on all data: {y_pred}")
    print(f"Actual values: {y.values}")

    return model

# Predict expiry
def predict_expiry(model, manufacture_date, expiry_date, purchase_date, temp_recommended, temp_actual):
    temp_diff = temp_actual - temp_recommended
    purchase = pd.to_datetime(purchase_date)
    today = datetime.datetime.now()
    days_since_purchase = (today - purchase).days
    features = pd.DataFrame([[temp_diff, days_since_purchase]], 
                           columns=['temp_diff', 'days_since_purchase'])
    days_remaining = model.predict(features)[0]
    return max(1, min(days_remaining, 30))

if __name__ == "__main__":
    model = train_model()