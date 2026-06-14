import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="CLV Prediction Dashboard", layout="wide")

# Load Data and Model
@st.cache_data
def load_data():
    df = pd.read_csv('processed_customer_data.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('rf_model.pkl')
    return model

df = load_data()
model = load_model()

# Sidebar
st.sidebar.title("CLV Dashboard")
st.sidebar.markdown("Predict and Segment Customers based on their lifetime value.")

# Main Title
st.title("Customer Lifetime Value (CLV) Prediction & Segmentation")

# View Selection
view = st.sidebar.selectbox("Select View", ["Executive Summary", "Customer Lookup", "Feature Insights"])

if view == "Executive Summary":
    st.header("📊 Executive Summary")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Avg Predicted CLV", f"${df['PredictedCLV'].mean():.2f}")
    col3.metric("VIP Customers", f"{len(df[df['Segment'] == 'VIP']):,}")
    col4.metric("Total Potential Revenue", f"${df['PredictedCLV'].sum():,.0f}")
    
    st.divider()
    
    # Charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Customer Segment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Segment', palette='viridis', ax=ax)
        st.pyplot(fig)
        
    with col_right:
        st.subheader("Revenue Contribution by Segment")
        segment_revenue = df.groupby('Segment')['PredictedCLV'].sum()
        fig, ax = plt.subplots()
        segment_revenue.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('viridis', 4), ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)

elif view == "Customer Lookup":
    st.header("🔍 Individual Customer Lookup")
    
    customer_id = st.selectbox("Select Customer ID", df['CustomerID'].unique())
    
    customer_data = df[df['CustomerID'] == customer_id].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Stats")
        st.write(f"**Recency:** {customer_data['Recency']} days")
        st.write(f"**Frequency:** {customer_data['Frequency']} orders")
        st.write(f"**Historical Spend:** ${customer_data['MonetaryValue']:.2f}")
        st.write(f"**Average Order Value:** ${customer_data['AOV']:.2f}")
        st.write(f"**Tenure:** {customer_data['TenureDays']} days")
        
    with col2:
        st.subheader("Prediction & Strategy")
        st.success(f"**Predicted Future CLV:** ${customer_data['PredictedCLV']:.2f}")
        st.info(f"**Assigned Segment:** {customer_data['Segment']}")
        
        # Strategy Logic
        strategy = ""
        if customer_data['Segment'] == 'VIP':
            strategy = "Invite to Exclusive Rewards Program and provide Premium Support."
        elif customer_data['Segment'] == 'High Value':
            strategy = "Target with Upselling and Loyalty Incentives."
        elif customer_data['Segment'] == 'Mid Value':
            strategy = "Send Personalized Product Recommendations."
        else:
            strategy = "Trigger Automated Re-engagement Discount Campaign."
            
        st.warning(f"**Recommended Action:** {strategy}")

elif view == "Feature Insights":
    st.header("💡 Feature Importance")
    st.write("Understand which customer behaviors drive future value.")
    
    # Feature Importance Plot
    features = ['Recency', 'Frequency', 'MonetaryValue', 'AOV', 'TenureDays']
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='magma', ax=ax)
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Takeaways:
    * **Monetary Value** is often the strongest predictor. Past spending is a great indicator of future spending.
    * **Recency** tells us how active the customer is. Fresh customers are more likely to return.
    * **Tenure** shows customer loyalty over time.
    """)

# Footer
st.divider()
st.caption("Built for Portfolio - Saif Eldeen | Data: Online Retail UCI")