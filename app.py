

import streamlit as st
import numpy as np
import pickle

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #00C9A7;
    }
    </style>
    """,
    unsafe_allow_html=True
    )


# load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🏦 Loan Approval Prediction System")

st.write("Enter your details to check loan approval")
st.markdown("### 📊 About This Project")
st.write("This web application uses Machine Learning to predict whether a loan will be approved or not based on user details such as income, credit history, and other factors.")

st.info("ℹ Please Fill in all details carefully. Use ℹ icons to understand each field.")


# inputs
gender = st.selectbox(
    "Gender",
    ["Male", "Female"],
    help="Applicant's gender"
)
married = st.selectbox(
    "Married",
    ["Yes", "No"],
    help="Marital status of the applicant"
)
dependents = st.selectbox(
    "Dependents",
    [0, 1, 2, "3+"],
    help="Number of family members dependent on you (3+ means 3 or more)"
)
education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"],
    help="Education level of the applicant"
)
self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"],
    help="Whether the applicant is self-employed or not"
)
income = st.number_input(
    "Applicant Income (monthly in ₹)",
    help="Monthly income of the applicant in Indian Rupees"
)
co_income = st.number_input(
    "Coapplicant Income (monthly in ₹)",
    help="Monthly income of co-applicant (family/spouse)"
)
loan_amount = st.number_input(
    "Loan Amount (in thousands ₹)",
    help="Loan amount in thousands. Example: 1 = ₹1 thousand, 100 = ₹100 thousand (₹1 lakh)"
)
loan_term = st.number_input(
    "Loan Term (in months)",
    help="Duration to repay loan (e.g., 12 = 1 year, 36 = 3 years, 60 = 5 years)"
)
credit_history = st.selectbox(
    "Credit History",
    [1, 0],
    help="1 = Good credit history, 0 = Bad credit history"
)
property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"],
    help="Location of the property"
)

# convert inputs
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

if dependents == "3+":
    dependents = 3

# prediction
if st.button("Predict"):
    input_data = np.array([[gender, married, dependents, education,
                            self_employed, income, co_income,
                            loan_amount, loan_term, credit_history,
                            property_area]])

    result = model.predict(input_data)
    prob = model.predict_proba(input_data)

    if result[0] == 1:
       st.success("Loan Approved ✅")
       st.write("Approval Probability:", prob[0][1])

       st.info("""
       ✔ The model predicts that your loan is likely to be approved.
       This is based on factors like good income, credit history, and repayment ability.
       """)

    else:
       st.error("Loan Not Approved ❌")
       st.write("Approval Probability:", prob[0][1])

       reasons = []

       if income < 3000:
           reasons.append("Low income")

       if loan_amount > income:
           reasons.append("Loan amount is high compared to income")

       if credit_history == 0:
           reasons.append("Poor or missing credit history")

       if dependents >= 3:
           reasons.append("High number of dependents")

       if len(reasons) == 0:
           reasons.append("Model decision is based on learned patterns from training data")

       st.warning("⚠ The model predicts that your loan may not be approved.")

       st.write("Possible reasons:")
       for r in reasons:
           st.write("-", r)

       
