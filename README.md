# Loan Approval Prediction System

##  What This Project Does

This project is a Machine Learning based web application that predicts whether a loan will be approved or not.

The user enters details like income, loan amount, credit history, and other information. Based on this input, the system gives:

* Loan Approved / Not Approved
* Probability of approval
* Explanation of the result
* Suggestions for improvement

---

## ⚙️ How to Set Up This Project (Step-by-Step)

Follow each step carefully. given below

---

### Step 1: Download the Project

* Click on **Code → Download ZIP**
* Extract the ZIP file
* Open the extracted folder

---

### Step 2: Open the Project in VS Code

* Open **VS Code**
* Click **File → Open Folder**
* Select your project folder

Make sure these files are present:

```
app.py
loan_model_training.py
model.pkl
train.csv
requirements.txt
```

---

### Step 3: Open Terminal and Install Libraries

* In VS Code, click **Terminal → New Terminal**
* A terminal will open at the bottom

You do not need to open any file for this step

Now type:

```
python -m pip install -r requirements.txt
```

---

### Step 4: Train the Model

In the same terminal, type:

```
python loan_model_training.py
```

This will:

* Train the model
* Create `model.pkl`

---

### Step 5: Run the Application

In the same terminal, type:

```
python -m streamlit run app.py
```

---

### Step 6: Open the Application

* The app will open automatically in your browser

 If not, open any browser and type:

```
http://localhost:8501
```

 If it still does not open:

* Close VS Code
* Open it again
* Repeat Steps 2 to Step 5

---

## How to Use the Application

1. Enter all required details:

* Gender → Male / Female
* Married → Yes / No
* Dependents → 0, 1, 2, or 3+
* Education → Graduate / Not Graduate
* Self Employed → Yes / No
* Applicant Income → Monthly income in ₹
* Coapplicant Income → Monthly income in ₹
* Loan Amount → In thousands (100 = ₹1 lakh)
* Loan Term → In months
* Credit History → 1 (Good) / 0 (Poor)
* Property Area → Rural / Semiurban / Urban

2. Click on **Predict**

---

## 📊 What You Will See

###  If Loan is Approved

* Message: **Loan Approved ✅**
* Probability value will be shown
* A short message explaining approval

---

###  If Loan is Not Approved

* Message: **Loan Not Approved ❌**
* Probability value will be shown

You will also see:

####  Possible Reasons:

* Low income
* High loan amount
* Poor credit history
* Too many dependents

#### Suggestions:

* Increase income or add co-applicant
* Reduce loan amount
* Maintain good credit history

---

## Limitations

* The model works only on dataset patterns
* It does NOT consider:

  * Loan purpose
  * Job stability
  * Real bank policies

 So predictions may be different from real-world decisions.

---

## 📁 Project Structure

```
├── report/
├── screenshots/
├── README.md
├── app.py
├── loan_model_training.py
├── model.pkl
├── requirements.txt
├── train.csv


```
