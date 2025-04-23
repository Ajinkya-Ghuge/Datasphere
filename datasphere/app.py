from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
from model import train_model, predict_expiry


app = Flask(__name__)

# Load or train ML model
model = train_model()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    category = request.form['category']
    product = request.form['product']
    manufacture_date = request.form['manufacture_date']
    expiry_date = request.form['expiry_date']
    purchase_date = request.form['purchase_date']
    temp_recommended = float(request.form['temp_recommended'])
    temp_actual = float(request.form['temp_actual'])

    # ML prediction
    days_remaining = predict_expiry(model, manufacture_date, expiry_date, purchase_date, temp_recommended, temp_actual)
    new_expiry = pd.to_datetime(purchase_date) + pd.Timedelta(days=days_remaining)
    savings = 2000  # Dummy savings for now (e.g., ₹2,000/milk batch)

    # Log to SQLite
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions 
                 (category TEXT, product TEXT, expiry TEXT, savings REAL)''')
    c.execute('INSERT INTO predictions VALUES (?, ?, ?, ?)', 
              (category, product, new_expiry.strftime('%Y-%m-%d'), savings))
    conn.commit()
    conn.close()

    return render_template('predict.html', 
                          category=category, 
                          product=product, 
                          expiry=new_expiry.strftime('%Y-%m-%d'), 
                          savings=savings)

# API endpoint
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    manufacture_date = data['manufacture_date']
    expiry_date = data['expiry_date']
    purchase_date = data['purchase_date']
    temp_recommended = float(data['temp_recommended'])
    temp_actual = float(data['temp_actual'])

    days_remaining = predict_expiry(model, manufacture_date, expiry_date, purchase_date, temp_recommended, temp_actual)
    new_expiry = pd.to_datetime(purchase_date) + pd.Timedelta(days=days_remaining)
    savings = 2000  # Dummy savings

    return jsonify({'expiry': new_expiry.strftime('%Y-%m-%d'), 'savings': savings})

if __name__ == '__main__':
    app.run(debug=True)