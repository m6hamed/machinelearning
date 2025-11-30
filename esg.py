import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ESG Revenue Predictor", layout="wide")

st.title("📊 Corporate Revenue Prediction Dashboard")
st.markdown("""
This app loads a pre-trained **Random Forest** model to predict a company's revenue based on its 
**Financials**, **ESG Scores**, and **Operational Metrics**.
""")

# --- 1. LOAD ASSETS (Model, Features, and Data for Dropdowns) ---
@st.cache_resource
def load_assets():
    # Load the Pre-trained Model
    try:
        with open('model.joblib', 'rb') as file:
            model = joblib.load(file)
    except FileNotFoundError:
        st.error("Error: 'model.joblib' not found.")
        return None, None, None

    # Load the Feature Names
    try:
        with open('feature_names.pkl', 'rb') as file:
            feature_names = pickle.load(file)
    except FileNotFoundError:
        st.error("Error: 'feature_names.pkl' not found.")
        return None, None, None


    
    df_raw = None
    
    try:
        
        df_raw = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\esg project\\archive (4).zip')
    except FileNotFoundError:
        try:
            
            
            df_raw = pd.read_csv('archive (4).zip')
            st.toast("Data loaded from 'archive (4).zip'", icon="📦")
        except FileNotFoundError:
            try:
                # Try 3: Look for generic zip name
                df_raw = pd.read_csv('archive.zip')
                st.toast("Data loaded from 'archive.zip'", icon="📦")
            except FileNotFoundError:
                 # Debug info to help you
                current_dir = os.getcwd()
                st.error(f"❌ Could not find data.")
                st.write(f"I am looking in this folder: `{current_dir}`")
                st.write("I tried looking for: `company_esg_financial_dataset.csv`, `archive (4).zip`, and `archive.zip`.")
                st.warning("Please verify the file is in this folder and spelled correctly.")
                
                # Create an empty fallback so the app doesn't crash
                df_raw = pd.DataFrame(columns=['Industry', 'Region', 'Revenue'])

    return model, feature_names, df_raw

# Execute loading
model, model_columns, df_raw = load_assets()

# Stop execution if model didn't load
if model is None:
    st.stop()

# --- 2. SIDEBAR: USER INPUTS ---
st.sidebar.header("📝 Input Company Parameters")

def user_input_features():
    # Financials
    market_cap = st.sidebar.number_input("Market Cap ($M)", min_value=10.0, value=5000.0)
    growth_rate = st.sidebar.slider("Growth Rate (%)", min_value=-20.0, max_value=50.0, value=3.5)
    
    # ESG Scores
    st.sidebar.subheader("ESG Scores")
    esg_env = st.sidebar.slider("Environmental Score", 0, 100, 70)
    esg_soc = st.sidebar.slider("Social Score", 0, 100, 60)
    esg_gov = st.sidebar.slider("Governance Score", 0, 100, 65)
    
    # Operations
    st.sidebar.subheader("Operational Metrics")
    energy = st.sidebar.number_input("Energy Consumption", value=50000.0)
    carbon = st.sidebar.number_input("Carbon Emissions", value=20000.0)
    water = st.sidebar.number_input("Water Usage", value=10000.0)
    
    # Categorical Selection
    # Safety check if dataframe is empty
    if not df_raw.empty and 'Industry' in df_raw.columns:
        industry_options = df_raw['Industry'].unique()
    else:
        industry_options = ['Technology', 'Retail', 'Finance']
        
    if not df_raw.empty and 'Region' in df_raw.columns:
        region_options = df_raw['Region'].unique()
    else:
        region_options = ['North America', 'Asia', 'Europe']
    
    industry = st.sidebar.selectbox("Industry", industry_options)
    region = st.sidebar.selectbox("Region", region_options)
    
    # Store in dictionary
    data = {
        'MarketCap': market_cap,
        'GrowthRate': growth_rate,
        'ESG_Environmental': esg_env,
        'ESG_Social': esg_soc,
        'ESG_Governance': esg_gov,
        'EnergyConsumption': energy,
        'CarbonEmissions': carbon,
        'WaterUsage': water,
        'Industry': industry,
        'Region': region,
        'ESG_Balance_Std': np.std([esg_env, esg_soc, esg_gov])
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display User Input
st.subheader("Your Input Configuration")
st.write(input_df)

# --- 3. PREDICTION LOGIC ---
if st.button("🚀 Predict Revenue"):
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_encoded)
    
    st.success(f"💰 Predicted Annual Revenue: **${prediction[0]:,.2f}**")
    
    # --- 4. EXPLAINABILITY ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Comparison to Industry Average")
        if not df_raw.empty:
            # Safe filter
            industry_data = df_raw[df_raw['Industry'] == input_df['Industry'][0]]
            if not industry_data.empty:
                industry_avg = industry_data['Revenue'].mean()
                
                fig, ax = plt.subplots()
                bars = ['Predicted', 'Industry Avg']
                values = [prediction[0], industry_avg]
                colors = ['#4CAF50', '#FFC107']
                ax.bar(bars, values, color=colors)
                ax.set_ylabel("Revenue")
                st.pyplot(fig)
            else:
                st.info("No historical data for this industry.")
        else:
            st.warning("Cannot compare: Source data not loaded.")
        
    with col2:
        st.subheader("Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-5:] 
            
            fig2, ax2 = plt.subplots()
            ax2.barh(range(len(indices)), importances[indices], align='center', color='#2196F3')
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels([model_columns[i] for i in indices])
            ax2.set_xlabel('Relative Importance')
            st.pyplot(fig2)