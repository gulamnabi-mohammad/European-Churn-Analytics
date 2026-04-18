import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="European Banking Churn Analytics", layout="wide")

st.title("🏦 Customer Segmentation & Churn Pattern Analytics")
st.markdown("**European Banking | Data Science Internship Project**")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/churn_data.csv")
    df.drop(columns=['Surname'], inplace=True)

    # Segmentation
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,30,45,60,100],
                           labels=['<30','30-45','46-60','60+'])

    df['CreditBand'] = pd.cut(df['CreditScore'], bins=[0,580,700,850],
                             labels=['Low','Medium','High'])

    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[-1,2,5,10],
                              labels=['New','Mid-term','Long-term'])

    df['BalanceSegment'] = pd.cut(df['Balance'],
                                 bins=[-1,0,50000,200000,df['Balance'].max()],
                                 labels=['Zero','Low','Medium','High'])

    return df

df = load_data()

# -----------------------------
# Train ML Model
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df[['CreditScore','Age','Balance','NumOfProducts','IsActiveMember']]
    y = df['Exited']
    model = LogisticRegression()
    model.fit(X, y)
    return model

model = train_model(df)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔎 Filter Customers")

geo_filter = st.sidebar.multiselect(
    "Geography",
    options=list(df['Geography'].unique()),
    default=list(df['Geography'].unique())
)

age_filter = st.sidebar.multiselect(
    "Age Group",
    options=list(df['AgeGroup'].dropna().unique()),
    default=list(df['AgeGroup'].dropna().unique())
)

balance_filter = st.sidebar.multiselect(
    "Balance Segment",
    options=list(df['BalanceSegment'].dropna().unique()),
    default=list(df['BalanceSegment'].dropna().unique())
)

filtered_df = df[
    (df['Geography'].isin(geo_filter)) &
    (df['AgeGroup'].isin(age_filter)) &
    (df['BalanceSegment'].isin(balance_filter))
]

# -----------------------------
# KPIs
# -----------------------------
st.subheader("📊 Key Performance Indicators")

overall_churn = filtered_df['Exited'].mean() * 100

high_value_threshold = filtered_df['Balance'].quantile(0.75)
high_value_df = filtered_df[filtered_df['Balance'] >= high_value_threshold]
high_value_churn = high_value_df['Exited'].mean() * 100

revenue_risk = high_value_df[high_value_df['Exited'] == 1]['Balance'].sum()

inactive_churn = filtered_df[
    filtered_df['IsActiveMember'] == 0
]['Exited'].mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Overall Churn Rate", f"{overall_churn:.2f}%")
col2.metric("High-Value Churn Rate", f"{high_value_churn:.2f}%")
col3.metric("Revenue at Risk (€)", f"{revenue_risk:,.0f}")
col4.metric("Inactive Customer Churn", f"{inactive_churn:.2f}%")

# -----------------------------
# Geography Chart
# -----------------------------
st.subheader("🌍 Churn by Geography")

geo_churn = filtered_df.groupby('Geography')['Exited'].mean().reset_index()
geo_churn['Exited'] *= 100

fig_geo = px.bar(geo_churn, x='Geography', y='Exited',
                 title="Geographic Churn Comparison",
                 labels={'Exited': 'Churn Rate (%)'})

st.plotly_chart(fig_geo, use_container_width=True)

# -----------------------------
# Age & Tenure Charts
# -----------------------------
st.subheader("👥 Demographic Analysis")

col1, col2 = st.columns(2)

age_churn = filtered_df.groupby('AgeGroup')['Exited'].mean().reset_index()
age_churn['Exited'] *= 100

fig_age = px.bar(age_churn, x='AgeGroup', y='Exited',
                 title="Churn by Age Group")

col1.plotly_chart(fig_age, use_container_width=True)

tenure_churn = filtered_df.groupby('TenureGroup')['Exited'].mean().reset_index()
tenure_churn['Exited'] *= 100

fig_tenure = px.bar(tenure_churn, x='TenureGroup', y='Exited',
                    title="Churn by Tenure")

col2.plotly_chart(fig_tenure, use_container_width=True)

# -----------------------------
# Balance Distribution
# -----------------------------
st.subheader("💎 Balance vs Churn")

fig_balance = px.box(filtered_df, x='Exited', y='Balance',
                     title="Balance Distribution")

st.plotly_chart(fig_balance, use_container_width=True)

# -----------------------------
# 🤖 Churn Prediction Section
# -----------------------------
st.markdown("---")
st.header("🤖 Predict Customer Churn")

age = st.number_input("Age", 18, 100, 30)
balance = st.number_input("Balance", 0.0)
products = st.selectbox("Products", [1,2,3,4])
active = st.selectbox("Active Member", [0,1])

if st.button("Predict"):
    prob = model.predict_proba([[650, age, balance, products, active]])
    churn_prob = prob[0][1] * 100

    st.write(f"Churn Probability: {churn_prob:.2f}%")

    if churn_prob > 60:
        st.error("High Risk 🚨")
    else:
        st.success("Low Risk ✅")

# -----------------------------
# Business Insights
# -----------------------------
st.markdown("---")
st.header("💡 Business Insights")

if inactive_churn > overall_churn:
    st.warning("Inactive users are more likely to churn → Increase engagement")

if high_value_churn > overall_churn:
    st.warning("High-value customers are leaving → Provide premium benefits")

# -----------------------------
# Raw Data
# -----------------------------
with st.expander("📄 View Data"):
    st.dataframe(filtered_df)
