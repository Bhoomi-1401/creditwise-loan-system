import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go

st.set_page_config(page_title="CreditWise Loan System", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .main-title { font-size: 40px; font-weight: bold; color: #00d4ff; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏦 CreditWise Loan Approval System</p>', unsafe_allow_html=True)
st.markdown("Fill in the details below and click **Predict** to check loan approval.")

@st.cache_data
def load_model():
    df = pd.read_csv("2 loan_approval_data.csv")
    df.drop("Applicant_ID", axis=1, inplace=True)
    numerical_cols = df.select_dtypes(include=["float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])
    obj_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = obj_imp.fit_transform(df[categorical_cols])
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])
    cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(cols, axis=1, inplace=True)
    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    return nb, scaler, ohe, X.columns.tolist()

model, scaler, ohe, feature_cols = load_model()

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 👤 Applicant Information")
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, max_value=20000, value=15000)
    coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, max_value=20000, value=0)
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    credit_score = st.number_input("Credit Score", min_value=550, max_value=799, value=750)
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed"])
    education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    st.markdown("### 🏠 Loan & Property Details")
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=40000, value=10000)
    loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=120)
    dti_ratio = st.slider("DTI Ratio", 0.0, 0.6, 0.1)
    savings = st.number_input("Savings ($)", min_value=0, max_value=20000, value=15000)
    collateral_value = st.number_input("Collateral Value ($)", min_value=0, max_value=40000, value=15000)
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

st.markdown("### 📋 Other Details")
col3, col4 = st.columns(2)
with col3:
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Home", "Education", "Business"])
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
with col4:
    employer_category = st.selectbox("Employer Category", ["Government", "Private", "NGO"])
    existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)

st.markdown("---")
predict_btn = st.button("🔍 Predict Loan Approval", use_container_width=True)

if predict_btn:
    input_dict = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": 1 if education_level == "Graduate" else 0,
    }
    ohe_input = ohe.transform([[employment_status, marital_status, loan_purpose, property_area, gender, employer_category]])
    ohe_df = pd.DataFrame(ohe_input, columns=ohe.get_feature_names_out(["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]))
    input_df = pd.DataFrame([input_dict])
    input_df = pd.concat([input_df, ohe_df], axis=1)
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={"text": "Loan Eligibility Score", "font": {"color": "white", "size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": "#00d4ff"},
            "steps": [
                {"range": [0, 40], "color": "#ff4444"},
                {"range": [40, 70], "color": "#ffaa00"},
                {"range": [70, 100], "color": "#00cc44"},
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="#0e1117", font_color="white", height=350)
    st.plotly_chart(fig, use_container_width=True)

    if prediction == 1:
        st.success(f"✅ Loan APPROVED! Eligibility Score: {probability:.1f}%")
    else:
        st.error(f"❌ Loan REJECTED! Eligibility Score: {probability:.1f}%")