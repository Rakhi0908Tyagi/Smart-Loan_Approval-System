#==================================
#IMPORT REQUIRED LIBRARIES
#==================================
import pandas as pd


#==================================
# LOAD DATASET
#==================================
# Reading the dataset from CSV file 
data = pd.read_csv('train.csv')

# Display first 5 rows of the dataset to understand its structure
print(data.head())


#==================================
# HANDLE MISSING VALUES
#==================================
# Filling missing values using mode ( most frequent value)

data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])

# Filling numerical columns with mean value

data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

#==================================
# CHECK IF ANY MISSING VALUES LEFT
#==================================
print(data.isnull().sum())


#==========================================
# CONVERT CATEGORICAL DATA INTO NUMERICAL
#==========================================

# Covert Gender column: Male= 1, Female= 0
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Convert Married column: Yes= 1, No= 0
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})

# Convert Education column: Graduate= 1, Not Graduate= 0
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})

# Convert Self_Employed column: Yes= 1, No= 0
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})

# Convert Property_Area column: Urban= 2, Semiurban= 1, Rural= 0
data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})

# Convert Loan_Status column: Y= 1, N= 0 (target variable)
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})


#===========================================
# HANDLE DEPENDENTS COLUMN (3+ to 3)
#===========================================

# Replace '3+' with 3 (because model needs numerical values)
data['Dependents'] = data['Dependents'].replace('3+', 3)

# Convert Dependents column to integer type
data['Dependents'] = data['Dependents'].astype(int)

# Check unique values in Dependents column to confirm conversion
print("Unique values in Dependents:", data['Dependents'].unique())


#===========================================
# SPLIT DATA INTO INPUT(X) AND OUTPUT(Y)
#===========================================

# X contains all features (independent variables) )
X = data.drop(columns=['Loan_ID', 'Loan_Status'])
# Y contains target variable (dependent variable, what we want to predict)
Y = data['Loan_Status']

#============================================
# SPLIT DATA INTO TRAINING AND TESTING SETS
#============================================
from sklearn.model_selection import train_test_split


# 80% training data, 20% testing data
# random_state=2 ensures that we get the same split every time we run the code (for reproducibility)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Print shape to understand split of data
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


#===================================================
# TRAIN MACHINE LEARNING MODEL (LOGISTIC REGRESSION)
#==================================================
from sklearn.linear_model import LogisticRegression


# Create model object with max_iter=1000 to ensure convergence 
# (default is 100, which may not be enough for this dataset)
model = LogisticRegression(max_iter=1000)

# Train model using training data
model.fit(X_train, Y_train)

#===========================================
# MODEL EVALUATION
#===========================================
from sklearn.metrics import accuracy_score

# prediction on training data
X_train_prediction = model.predict(X_train)

# calculate training accuracy
train_accuracy = accuracy_score(Y_train, X_train_prediction)

print("Training Accuracy:",train_accuracy)


# prediction on testing data
X_test_prediction = model.predict(X_test)

# calculate testing accuracy
test_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Testing Accuracy:", test_accuracy)


# ==============================
# SAVE TRAINED MODEL
# ==============================

import pickle

# Save model to file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully as model.pkl")



