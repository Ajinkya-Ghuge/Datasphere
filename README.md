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
* 🧾 **Minimize Resource Waste:** Assists production facilities in mitigating supply chain decay through preventative timelines.
* 🛡️ **Optimize Safety Assurance:** Provides chemical, pharmaceutical, and consumable goods providers with predictive safety safety nets.
* 📦 **Streamline Inventory Flow:** Equips warehouse managers with accurate predictive indices for first-expired, first-out (FEFO) strategies.

---

## ✨ Key Features

* 🔍 **Multi-Variable Feature Analysis:** Aggregates multi-dimensional product configurations (e.g., compounding components, air exposure profiles, baseline temperature bounds).
* 🧠 **Advanced Regression Inference:** Uses trained scikit-learn estimators to convert dynamic environmental thresholds into real-time shelf life counts.
* 📊 **Insightful Exploratory Dashboards:** Packaged alongside exploratory rendering matrices utilizing Matplotlib and Seaborn for dataset profiling.
* 🔌 **API Integration Layer:** Exposes modular HTTP service routes designed to accept payload vectors and stream instantaneous predictions out to client ERP nodes.

---

## 🏗️ System Architecture

### Component Diagram

```mermaid
graph TD
    %% Styling configurations
    classDef client fill:#1a233a,stroke:#00f5ff,stroke-width:2px,color:#fff;
    classDef backend fill:#0d1428,stroke:#ff00aa,stroke-width:2px,color:#fff;
    classDef service fill:#112244,stroke:#ffd700,stroke-width:2px,color:#fff;
    classDef storage fill:#1c1c24,stroke:#00ff88,stroke-width:2px,color:#fff;

    subgraph ClientLayer [User Web Interface Layer]
        A[HTML Input Dashboards<br>Product Parameter Form UI]:::client
        B[Static App Assets<br>Form Validation Engine]:::client
        A <--> B
    end

    subgraph ServerLayer [Flask Application Backend]
        C[app.py<br>HTTP Route Handlers & Interface]:::backend
    end

    subgraph CoreEngine [Predictive AI Layer]
        D[model.py / predictor<br>scikit-learn Inference Models]:::service
    end

    subgraph AnalyticsLayer [Data Visualization Space]
        E[Jupyter Notebooks<br>EDA Data Exploration Scripts]:::storage
    end

    %% Structural Connectivity Flow
    B <=>|REST API JSON Payloads| C
    C <=>|Vector Extrapolation| D
    E -->|Model Performance Validation| D
Request Flow Workflow
Code snippet
sequenceDiagram
    autonumber
    actor User as Inventory Admin Panel
    participant App as Flask Router (app.py)
    participant Model as AI Core Engine (model.py)
    participant Metrics as Visualization Layer

    User->>App: Submits Product Specs (Ingredients, Temperature, Packaging)
    App->>Model: Maps Incoming Fields to Data Wrangling NumPy Matrix Array
    
    activate Model
    Model->>Model: Applies Preprocessing Transformers & Evaluation Pipelines
    Model-->>App: Dispatched Predicted Fractional Expiry & Target Timelines
    deactivate Model

    App->>Metrics: Sends Structural Run Event Context Log Coordinates
    App-->>User: Populates Dynamic UI Cards with Precise Expiry Inferences
🤖 Machine Learning Pipeline
The internal intelligence matrix transforms raw physical components into highly accurate storage decay curves:

Code snippet
graph LR
    classDef step fill:#112244,stroke:#00f5ff,stroke-width:2px,color:#fff;
    
    A[Data Ingestion<br>Ingredients & Storage Profiles]:::step --> B[Data Wrangling<br>Pandas & NumPy Cleaning]:::step
    B --> C[Feature Matrix Encoding<br>Packaging & Compound Encoding]:::step
    C --> D[Model Estimator Pipeline<br>Scikit-Learn Evaluation]:::step
    D --> E[Inference Stream Outputs<br>Target Expiry Date Projections]:::step
🚀 Project Workflow
Code snippet
flowchart TD
    Start([Initialize Predictive Server Application]) --> Ingest[Parse Configuration & Load ML Weights]
    Ingest --> ServerCheck{Bind Web Sockets & Paths}
    
    ServerCheck --|Port Busy| Kill[Terminate Dead Connections] --> Ingest
    ServerCheck --|Available| Boot[Serve Local Flask Service Host]
    
    Boot --> Listen{Awaiting Evaluation Request}
    Listen -->|Form Ingestion Submitted| Extrapolate[Wrangle Payload Fields via Pandas Vectors]
    Extrapolate --> RunInference[Evaluate Profile Against Regression Pipeline]
    RunInference --> OutputJSON[Generate Response Mapping Context Data]
    OutputJSON --> ReturnView[Render Results Card View in Dashboard Panel]
    
    ReturnView --> End([Present Active Expiry Profile to User Interface])
