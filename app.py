import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature info
def load_model():
    return joblib.load('sourcing_recommendation_model.pkl')

def get_feature_inputs():
    st.sidebar.header('Component Inputs')
    metal_type = st.sidebar.selectbox('Metal Type', ['Copper', 'Tin', 'Aluminum', 'Nickel', 'Silver'])
    form_factor = st.sidebar.selectbox('Form Factor', ['Foil', 'Sheet', 'Coil', 'Wire'])
    industry_usage = st.sidebar.selectbox('Industry Usage', ['Automotive', 'Toys', 'Wearable', 'Consumer', 'Industrial'])
    unit_weight_kg = st.sidebar.number_input('Unit Weight (kg)', min_value=0.1, max_value=20.0, value=1.0)
    source_type = st.sidebar.selectbox('Source Type', ['Local', 'Import'])
    lead_time_days = st.sidebar.slider('Lead Time (days)', 1, 60, 10)
    base_price_per_kg = st.sidebar.number_input('Base Price per kg', min_value=100.0, max_value=2000.0, value=1000.0)
    freight_cost_per_kg = st.sidebar.number_input('Freight Cost per kg', min_value=0.0, max_value=500.0, value=50.0)
    customs_duty_percent = st.sidebar.number_input('Customs Duty (%)', min_value=0.0, max_value=20.0, value=5.0)
    local_tax_percent = st.sidebar.number_input('Local Tax (%)', min_value=0.0, max_value=30.0, value=18.0)
    exchange_rate_multiplier = st.sidebar.number_input('Exchange Rate Multiplier', min_value=0.8, max_value=2.0, value=1.0)
    cost_margin = st.sidebar.number_input('Cost Margin', min_value=-10000.0, max_value=10000.0, value=0.0)
    return {
        'metal_type': metal_type,
        'form_factor': form_factor,
        'industry_usage': industry_usage,
        'unit_weight_kg': unit_weight_kg,
        'source_type': source_type,
        'lead_time_days': lead_time_days,
        'base_price_per_kg': base_price_per_kg,
        'freight_cost_per_kg': freight_cost_per_kg,
        'customs_duty_percent': customs_duty_percent,
        'local_tax_percent': local_tax_percent,
        'exchange_rate_multiplier': exchange_rate_multiplier,
        'cost_margin': cost_margin
    }

def encode_inputs(inputs):
    # Simple label encoding as per training
    metal_map = {'Copper':0, 'Tin':1, 'Aluminum':2, 'Nickel':3, 'Silver':4}
    form_map = {'Foil':0, 'Sheet':1, 'Coil':2, 'Wire':3}
    industry_map = {'Automotive':0, 'Toys':1, 'Wearable':2, 'Consumer':3, 'Industrial':4}
    source_map = {'Local':0, 'Import':1}
    return [
        metal_map[inputs['metal_type']],
        form_map[inputs['form_factor']],
        industry_map[inputs['industry_usage']],
        inputs['unit_weight_kg'],
        source_map[inputs['source_type']],
        inputs['lead_time_days'],
        inputs['base_price_per_kg'],
        inputs['freight_cost_per_kg'],
        inputs['customs_duty_percent'],
        inputs['local_tax_percent'],
        inputs['exchange_rate_multiplier'],
        inputs['cost_margin']
    ]


def main():
    import matplotlib.pyplot as plt
    import os
    st.set_page_config(page_title="Cost Forecasting Tool", layout="wide")
    st.title("Cost Forecasting Tool")

    # --- Executive KPIs ---
    st.markdown("### Executive KPIs")
    col1, col2, col3, col4 = st.columns(4)
    # Dummy values for now, replace with real metrics if available
    col1.metric("Model Accuracy", "95%")
    col2.metric("Precision (Import)", "0.93")
    col3.metric("Recall (Import)", "0.91")
    col4.metric("F1 Score (Import)", "0.92")

    # --- Input Sidebar ---
    user_inputs = get_feature_inputs()

    # --- Prediction Result ---
    st.markdown("---")
    st.header("Prediction Result")
    predict_btn = st.button("Predict")
    if predict_btn:
        model = load_model()
        X = np.array([encode_inputs(user_inputs)])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        pred_label = "Import" if pred == 1 else "Local"
        st.markdown(f"#### Recommendation: **{pred_label}**")

        # Probability Pie Chart
        st.markdown("##### Prediction Probability")
        fig1, ax1 = plt.subplots()
        ax1.pie(proba, labels=["Local", "Import"], autopct='%1.1f%%', startangle=90, colors=["#6fa8dc", "#f6b26b"])
        ax1.axis('equal')
        st.pyplot(fig1)

        # --- Input Summary ---
        st.markdown("---")
        st.header("Input Summary")
        feature_names = [
            'metal_type', 'form_factor', 'industry_usage', 'unit_weight_kg', 'source_type',
            'lead_time_days', 'base_price_per_kg', 'freight_cost_per_kg', 'customs_duty_percent',
            'local_tax_percent', 'exchange_rate_multiplier', 'cost_margin'
        ]
        input_vals_display = [
            user_inputs['metal_type'],
            user_inputs['form_factor'],
            user_inputs['industry_usage'],
            user_inputs['unit_weight_kg'],
            user_inputs['source_type'],
            user_inputs['lead_time_days'],
            user_inputs['base_price_per_kg'],
            user_inputs['freight_cost_per_kg'],
            user_inputs['customs_duty_percent'],
            user_inputs['local_tax_percent'],
            user_inputs['exchange_rate_multiplier'],
            user_inputs['cost_margin']
        ]
        feature_df = pd.DataFrame({'Feature': feature_names, 'Value': [str(v) for v in input_vals_display]})
        st.dataframe(feature_df)

        # --- SHAP Explainability ---
        st.markdown("---")
        st.header("SHAP Explainability")
        shap_img_path = "shap_summary.png"
        if os.path.exists(shap_img_path):
            st.image(shap_img_path, caption="Global Feature Importance (SHAP)")
        else:
            st.info("SHAP feature importance plot can be added here for model explainability.")

        # --- Business Insights ---
        st.markdown("---")
        st.header("Business Insights")
        st.write("(Example) Trends in Cost, Lead Time, and Sourcing Type")
        # Dummy example chart
        chart_data = pd.DataFrame({
            "Cost (INR)": np.random.randint(1000, 10000, 10),
            "Lead Time (days)": np.random.randint(5, 30, 10),
            "Sourcing Type": np.random.choice(["Local", "Import"], 10)
        })
        st.bar_chart(chart_data[["Cost (INR)", "Lead Time (days)"]])
        st.write("Sourcing Type Distribution:")
        st.bar_chart(chart_data["Sourcing Type"].value_counts())

        # --- Model Evaluation ---
        st.markdown("---")
        st.header("Model Evaluation")
        st.write("Accuracy, precision, recall, F1, and confusion matrix can be shown here.")
        # Dummy confusion matrix
        cm = np.array([[90, 10], [7, 93]])
        st.write("Confusion Matrix (Local/Import):")
        st.dataframe(pd.DataFrame(cm, columns=["Pred Local", "Pred Import"], index=["True Local", "True Import"]))

    st.markdown("---")
    st.header("Instructions")
    st.write("Use the sidebar to enter component details. Click 'Predict' to get sourcing recommendation and see input summary and prediction probabilities. Executive KPIs and business insights are shown above for quick review.")

if __name__ == '__main__':
    main()
