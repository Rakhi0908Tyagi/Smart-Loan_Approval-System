#======================================
# IMPORT LIBRARIES
#======================================
import streamlit as st
import numpy as np
import pickle

#======================================
# CUSTOM STYLING (UI DESIGN)
#======================================
# This section is used to customize the look of the app using CSS

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


#======================================
# LOAD TRAINED MODEL
#======================================
# load the saved machine learning model from file
model = pickle.load(open('model.pkl', 'rb'))

#======================================
# APP TITLE AND DESCRIPTION
#======================================
st.title("🏦 Loan Approval Prediction System")

st.write("Enter your details to check loan approval")

# About section to explain the project
st.markdown("### 📊 About This Project")
st.write("This web application uses Machine Learning to predict whether a loan will be approved or not based on user details such as income, credit history, and other factors.")

# Instruction message for users
st.info("ℹ Please Fill in all details carefully. Use ℹ icons to understand each field.")


#======================================
# USER INPUT SECTION
#======================================
#This section collects input data from the user 

#--------------------------------------
# CATEGORICAL INPUTS
#--------------------------------------

# Gender of applicant
gender = st.selectbox(
    "Gender",
    ["Male", "Female"],
    help="Select the Applicant's gender"
)

# Marital Status of applicant
married = st.selectbox(
    "Married",
    ["Yes", "No"],
    help="Select whether the applicant is married"
)

# Number of dependents
dependents = st.selectbox(
    "Dependents",
    [0, 1, 2, "3+"],
    help="Number of family members dependent on you (3+ means 3 or more)"
)

# Education level of applicant
education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"],
    help="Select the applicant's education level"
)

# Employment status of applicant
self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"],
    help="Select if the applicant is self-employed or not"
)

#--------------------------------------
# NUMERICAL INPUTS
#--------------------------------------

# Monthly income of applicant
income = st.number_input(
    "Applicant Income (monthly in ₹)",
    help="Enter the monthly income of the applicant in Indian Rupees"
)

# Monthly income of co-applicant
co_income = st.number_input(
    "Coapplicant Income (monthly in ₹)",
    help="Enter the monthly income of the co-applicant (family/spouse) if any, otherwise enter 0"
)

# Loan amount requested by applicant
loan_amount = st.number_input(
    "Loan Amount (in thousands ₹)",
    help="Loan amount in thousands. Example: 1 = ₹1 thousand, 100 = ₹100 thousand (₹1 lakh)"
)

# Loan term in months
loan_term = st.number_input(
    "Loan Term (in months)",
    help="Duration to repay loan (e.g., 12 = 1 year, 36 = 3 years, 60 = 5 years)"
)

# Credit history of applicant
credit_history = st.selectbox(
    "Credit History",
    [1, 0],
    help="1 = Good credit history, 0 = Bad credit history"
)

# Property area of applicant
property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"],
    help="Location of the property"
)


#==========================================
# DATA PREPROCESSING FOR MODEL INPUT
#==========================================
# Convert user input into numerical format for model

# Convert gender
gender = 1 if gender == "Male" else 0

# Convert martial status
married = 1 if married == "Yes" else 0

# Convert education
education = 1 if education == "Graduate" else 0

# Convert employment status
self_employed = 1 if self_employed == "Yes" else 0

# Convert property area
property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

# Convert dependents
if dependents == "3+":
    dependents = 3


#========================================
# PREDICTION SECTION
#========================================

if st.button("Predict"):

    # Create input array in correct format for model
    input_data = np.array([[gender, married, dependents, education,
                            self_employed, income, co_income,
                            loan_amount, loan_term, credit_history,
                            property_area]])
    

    # Make prediction
    result = model.predict(input_data)

    # Get probability
    prob = model.predict_proba(input_data)

    #===================================
    # RESULT DISPLAY SECTION
    #===================================

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

       #================================
       # REASON GENERATION 
       #================================
       reasons = []

       # Check income
       if income < 3000:
           reasons.append("Low income")
       
       # Check loan vs income 
       if loan_amount * 1000 > income * 10:
           reasons.append("Loan amount is high compared to income")
       
       # Check credit history
       if credit_history == 0:
           reasons.append("Poor or missing credit history")
    
       # Check dependents
       if dependents >= 3:
           reasons.append("High number of dependents")
       
       # Default reason
       if len(reasons) == 0:
           reasons.append("Model decision is based on learned patterns from training data")
       
       # Show explanation to user
       st.warning("⚠ The model predicts that your loan may not be approved.")

       st.write("Possible reasons:")
       for r in reasons:
           st.write("-", r)

       
