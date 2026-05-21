import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="FraudOps Dashboard", page_icon="🛡️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, 'fraud_model.pkl'))

@st.cache_data
def load_data():
    X = pd.read_csv(os.path.join(BASE_DIR, 'test_features.csv'))
    y = pd.read_csv(os.path.join(BASE_DIR, 'test_labels.csv'))
    
    df = X.copy()
    df['Actual_Fraud'] = y.values
    return df

try:
    model = load_model()
    df = load_data()
except FileNotFoundError:
    st.error("Error: Could not find 'fraud_model.pkl', 'test_features.csv', or 'test_labels.csv'. Please ensure they are in the same folder as app.py.")
    st.stop()

feature_cols = [c for c in df.columns if c not in ['TransactionID', 'Actual_Fraud']]
df['Risk_Score'] = model.predict_proba(df[feature_cols])[:, 1]
df['Risk_Tier'] = np.select(
    [df['Risk_Score'] >= 0.75, df['Risk_Score'] >= 0.40], 
    ['🔴 Critical', '🟡 Suspicious'], 
    default='🟢 Clear'
)

st.sidebar.title("FraudOps Navigation")
page = st.sidebar.radio("Go to:", ["Overview", "Transaction Explorer", "SHAP Explainer"])

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

min_amt, max_amt = float(df['TransactionAmt'].min()), float(df['TransactionAmt'].max())
amt_filter = st.sidebar.slider("Transaction Amount Range ($)", min_amt, max_amt, (min_amt, max_amt))

filtered_df = df[(df['TransactionAmt'] >= amt_filter[0]) & (df['TransactionAmt'] <= amt_filter[1])]

if page == "Overview":
    st.title("System Overview")
    
    total_txns = len(filtered_df)
    total_fraud = filtered_df['Actual_Fraud'].sum()
    detection_rate = (filtered_df[(filtered_df['Actual_Fraud'] == 1) & (filtered_df['Risk_Score'] >= 0.75)].shape[0] / total_fraud) * 100 if total_fraud > 0 else 0
    avg_fraud_amt = filtered_df[filtered_df['Actual_Fraud'] == 1]['TransactionAmt'].mean() if total_fraud > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_txns:,}")
    col2.metric("Confirmed Fraud", f"{total_fraud:,}")
    col3.metric("Critical Detection Rate", f"{detection_rate:.1f}%")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amt:.2f}")
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Risk Tier Distribution")
        tier_counts = filtered_df['Risk_Tier'].value_counts().reset_index()
        tier_counts.columns = ['Risk_Tier', 'Count']
        fig1 = px.pie(tier_counts, values='Count', names='Risk_Tier', 
                      color='Risk_Tier', color_discrete_map={'🔴 Critical':'#e74c3c', '🟡 Suspicious':'#f1c40f', '🟢 Clear':'#2ecc71'})
        st.plotly_chart(fig1, width='stretch')
        
    with colB:
        st.subheader("Fraud Status vs. Transaction Amount")
        fig2 = px.box(filtered_df, x="Actual_Fraud", y="TransactionAmt", color="Actual_Fraud",
                      labels={"Actual_Fraud": "Is Fraud (0=No, 1=Yes)"})
        fig2.update_yaxes(range=[0, filtered_df['TransactionAmt'].quantile(0.95)]) 
        st.plotly_chart(fig2, width='stretch')

elif page == "Transaction Explorer":
    st.title("Transaction Explorer")
    st.markdown("Filter and search through recent transactions. Critical items require immediate review.")
    
    selected_tier = st.selectbox("Filter by Risk Tier", ["All", "🔴 Critical", "🟡 Suspicious", "🟢 Clear"])
    
    display_df = filtered_df.copy()
    if selected_tier != "All":
        display_df = display_df[display_df['Risk_Tier'] == selected_tier]
        
    display_df = display_df.sort_values(by="Risk_Score", ascending=False)
    
    cols_to_show = ['TransactionID', 'Risk_Tier', 'Risk_Score', 'TransactionAmt', 'HourOfDay', 'DeviceRisk', 'Actual_Fraud']
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]
    
    max_rows = 1000
    if len(display_df) > max_rows:
        st.info(f"Showing top {max_rows} highest-risk transactions out of {len(display_df)} total.")
        display_df = display_df.head(max_rows)
    
    st.dataframe(display_df[cols_to_show].style.background_gradient(subset=['Risk_Score'], cmap='Reds'), width='stretch')

elif page == "SHAP Explainer":
    st.title("Automated Decision Explainer")
    st.markdown("Enter a Transaction ID to understand exactly **why** the model generated its risk score.")
    
    txn_id_input = st.number_input("Enter Transaction ID:", min_value=int(df['TransactionID'].min()), max_value=int(df['TransactionID'].max()), step=1)
    
    if st.button("Generate Explanation"):
        txn_data = df[df['TransactionID'] == txn_id_input]
        
        if txn_data.empty:
            st.warning("Transaction ID not found.")
        else:
            features_only = txn_data[feature_cols]
            risk = txn_data['Risk_Score'].values[0]
            tier = txn_data['Risk_Tier'].values[0]
            
            st.subheader(f"Risk Score: {risk:.2f} ({tier})")
            
            with st.spinner('Calculating SHAP values...'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(features_only)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig)
                plt.clf() 
                
                st.markdown("### Analyst Briefing")
                top_feature_idx = np.argmax(np.abs(shap_values.values[0]))
                top_feature = feature_cols[top_feature_idx]
                top_feature_val = features_only[top_feature].values[0]
                
                if risk >= 0.75:
                    st.error(f"**Action Required:** This transaction is highly suspicious. The primary driver for this alert is **{top_feature}** (Value: {top_feature_val}), which heavily deviates from baseline normal behavior and pushes the fraud probability up.")
                elif risk >= 0.40:
                    st.warning(f"**Review Suggested:** This is a borderline transaction. The model detected mixed signals, with **{top_feature}** playing the largest role in the current score. Please review associated identity and device metrics.")
                else:
                    st.success(f"**Cleared:** This transaction aligns with legitimate behavior profiles. No anomalous spikes were detected in the core routing features.")