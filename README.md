<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=250&section=header&text=Expiry%20Date%20Predictor&fontSize=40&animation=fadeIn&fontAlignY=38" width="100%" alt="Project Banner" />
</p>

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Orbitron&weight=900&size=42&duration=3000&pause=1000&color=00F5FF&center=true&vCenter=true&width=900&lines=Expiry+Date+Predictor;AI-Powered+Shelf+Life+Intelligence" alt="Expiry Date Predictor" />

<br/>

### 🧪 Smart AI Shelf-Life Prediction Platform

**By [Ajinkya Ghuge](https://github.com/Ajinkya-Ghuge)**

*An AI-powered intelligence platform built to predict the expiry dates and shelf stability of food and pharmaceutical products using machine learning.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

<br/>

> *"No more manual tracking or guesswork. Just smart, accurate stability predictions based on data."*

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Project Overview](#-project-overview) |
| 2 | [Key Features](#-key-features) |
| 3 | [System Architecture](#-system-architecture) |
| 4 | [Machine Learning Pipeline](#-machine-learning-pipeline) |
| 5 | [Project Workflow](#-project-workflow) |
| 6 | [Tech Stack](#-tech-stack) |
| 7 | [Project Structure](#-project-structure) |
| 8 | [Installation & Local Setup](#-installation--local-setup) |
| 9 | [API Reference](#-api-reference) |

---

## 🎯 Project Overview

Manual tracking and guesswork regarding product expiration lead to massive inventory losses, safety risks, and operational waste. The **Expiry Date Predictor** acts as an automated solution that evaluates intricate raw feature variables—such as dynamic ingredients, volatile packaging materials, and active storage environment conditions—to forecast an exact, data-backed shelf life duration matrix.

### Core Objectives
- 🧾 **Minimize Resource Waste:** Assists production facilities in mitigating supply chain decay through preventative timelines.
- 🛡️ **Optimize Safety Assurance:** Provides chemical, pharmaceutical, and consumable goods providers with predictive safety safety nets.
- 📦 **Streamline Inventory Flow:** Equips warehouse managers with accurate predictive indices for first-expired, first-out (FEFO) strategies.

---

## ✨ Key Features

- 🔍 **Multi-Variable Feature Analysis:** Aggregates multi-dimensional product configurations (e.g., compounding components, air exposure profiles, baseline temperature bounds).
- 🧠 **Advanced Regression Inference:** Uses trained scikit-learn estimators to convert dynamic environmental thresholds into real-time shelf life counts.
- 📊 **Insightful Exploratory Dashboards:** Packaged alongside exploratory rendering matrices utilizing Matplotlib and Seaborn for dataset profiling.
- 🔌 **API Integration Layer:** Exposes modular HTTP service routes designed to accept payload vectors and stream instantaneous predictions out to client ERP nodes.

---

## 🏗️ System Architecture

### Component Diagram

```mermaid
flowchart TB

    subgraph ClientLayer["User Web Interface Layer"]
        A["HTML Input Dashboard<br/>Product Parameter Form UI"]
        B["Static Assets<br/>Validation Engine"]
    end

    subgraph ServerLayer["Flask Backend"]
        C["app.py<br/>HTTP Route Handlers"]
    end

    subgraph CoreEngine["Predictive AI Layer"]
        D["model.py<br/>ML Inference Engine"]
    end

    subgraph AnalyticsLayer["Analytics & Validation"]
        E["Jupyter Notebooks<br/>EDA & Model Evaluation"]
    end

    A --> B
    B -->|JSON Request| C
    C -->|Prediction Request| D
    D -->|Prediction Response| C
    E -->|Model Validation| D
    C -->|JSON Response| B
```

---

## 🔄 Request Flow

```mermaid
sequenceDiagram

    actor User

    participant Browser
    participant FlaskApp
    participant Model
    participant Analytics

    User->>Browser: Enter Product Information

    Browser->>FlaskApp: POST /predict

    FlaskApp->>Model: Preprocess Features

    activate Model

    Model->>Model: Feature Engineering
    Model->>Model: Run ML Prediction

    Model-->>FlaskApp: Predicted Shelf Life

    deactivate Model

    FlaskApp->>Analytics: Log Prediction Metrics

    Analytics-->>FlaskApp: Tracking Information

    FlaskApp-->>Browser: JSON Response

    Browser-->>User: Display Expiry Prediction
```

---

# 🤖 Machine Learning Pipeline

```mermaid
flowchart LR

    A["Data Collection"] -->
    B["Data Cleaning & Preprocessing"]

    B -->
    C["Feature Engineering"]

    C -->
    D["Model Training"]

    D -->
    E["Model Evaluation"]

    E -->
    F["Expiry Date Prediction"]
```

---

# 🚀 Project Workflow

```mermaid
flowchart TD

    Start([Start])

    --> Load[Load Model]

    Load --> Validate{Model Available?}

    Validate -->|No| Error[Raise Error]

    Validate -->|Yes| Run[Start Flask Server]

    Run --> Wait[Wait For User Request]

    Wait --> Input[Receive Product Details]

    Input --> Process[Preprocess Data]

    Process --> Predict[Generate Prediction]

    Predict --> Result[Create Response]

    Result --> Display[Show Prediction]

    Display --> End([Finish])
```

---

# 🛠️ Tech Stack

| Category         | Technology          | Purpose           |
| ---------------- | ------------------- | ----------------- |
| Language         | Python 3.8+         | Core Development  |
| Backend          | Flask               | Web Application   |
| Data Processing  | Pandas, NumPy       | Data Manipulation |
| Machine Learning | Scikit-Learn        | Prediction Models |
| Visualization    | Matplotlib, Seaborn | Analytics         |
| Experimentation  | Jupyter Notebook    | Model Training    |

---

# 📂 Project Structure

```plaintext
expiry-predictor/
│
├── data/
│   └── datasets/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│
├── notebooks/
│
├── app.py
├── model.py
├── constraint.py
├── requirements.txt
└── README.md
```

---

# ⚡ Installation & Local Setup

## Prerequisites

* Python 3.8+
* pip

## Clone Repository

```bash
git clone https://github.com/Ajinkya-Ghuge/Datasphere.git

cd Datasphere
```

## Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

---

# 🔌 API Reference

| Method | Endpoint | Description         |
| ------ | -------- | ------------------- |
| GET    | /        | Load Dashboard      |
| POST   | /predict | Predict Expiry Date |

### Example Request

```json
{
  "temperature": 25,
  "humidity": 60,
  "packaging": "Plastic",
  "ingredients": "Milk Powder"
}
```

### Example Response

```json
{
  "predicted_shelf_life_days": 365,
  "confidence_score": 0.94
}
```

---

<div align="center">

## 👤 Author

### Ajinkya Ghuge

GitHub: https://github.com/Ajinkya-Ghuge

⭐ Star this repository if you found it useful!

</div>
