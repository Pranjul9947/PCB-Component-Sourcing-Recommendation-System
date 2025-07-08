# PCB Component Sourcing Recommendation System

## Overview
This project provides a cost forecasting and sourcing recommendation system for PCB (Printed Circuit Board) components, helping businesses decide between Local and Import sourcing options. It leverages synthetic data generation, machine learning, and explainable AI to deliver actionable insights via an interactive Streamlit dashboard.

---

## Features
- **Synthetic Data Generation**: Simulates realistic sourcing scenarios for PCB components, including cost, lead time, taxes, and more.
- **Model Training & Explainability**: Trains a classification model to recommend sourcing type, with feature engineering, class balancing, hyperparameter tuning, and SHAP-based explainability.
- **Interactive Streamlit Dashboard**: Business-friendly UI for inputting component details, viewing recommendations, probabilities, feature importances, and business insights.
- **Business Insights & KPIs**: Visualizes trends, model performance, and key metrics for executive review.

---

## Project Structure
```
├── app.py                        # Streamlit dashboard
├── generate_sourcing_data.py     # Synthetic data generator
├── model_train_predict.py        # Model training & explainability
├── sourcing_recommendation_model.pkl # Trained model
├── synthetic_sourcing_data.csv   # Generated dataset
├── shap_summary.png              # SHAP feature importance plot
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Usage
### 1. Generate Synthetic Data
Run the data generator to create a dataset:
```bash
python generate_sourcing_data.py
```
This will output `synthetic_sourcing_data.csv`.

### 2. Train the Model
Train and evaluate the sourcing recommendation model:
```bash
python model_train_predict.py
```
This will output `sourcing_recommendation_model.pkl` and `shap_summary.png`.

### 3. Launch the Dashboard
Start the Streamlit app for interactive recommendations:
```bash
streamlit run app.py
```

---

## Streamlit Dashboard Overview
- **Sidebar**: Enter component details (metal type, form factor, industry, costs, taxes, etc.)
- **Main Area**:
  - **Prediction Result**: Sourcing recommendation (Local/Import) and probability pie chart
  - **Input Summary**: Review your inputs
  - **SHAP Explainability**: Visualize feature importances (global SHAP plot)
  - **Business Insights**: Trends in cost, lead time, and sourcing type
  - **Model Evaluation**: Key metrics (accuracy, precision, recall, F1, confusion matrix)
  - **Executive KPIs**: Quick stats for decision makers

---

## Business Insights
- **Cost vs. Lead Time**: Visualize how sourcing type impacts cost and delivery timelines.
- **Sourcing Distribution**: See the proportion of Local vs. Import recommendations.
- **Feature Importance**: Understand which factors most influence sourcing decisions (e.g., cost margin, lead time, taxes).
- **Model Performance**: Review accuracy, precision, recall, F1, and confusion matrix to assess reliability.

---

## Model Explainability
- **SHAP Summary Plot**: Shows global feature importance, helping users understand the model's decision logic.
- **Input Summary**: Lets users verify the data used for prediction.

---

## Requirements
See `requirements.txt` for all dependencies. Key packages include:
- streamlit
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- joblib
- matplotlib
- shap

---

## Notes
- The system uses synthetic data; for production, retrain with real business data.
- All business insights and KPIs are for demonstration and can be replaced with real metrics.
- The dashboard is designed for business users with no coding required.

---

## Contact
For questions or enhancements, please contact the project maintainer.
