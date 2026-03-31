# %%

# %%
import pandas as pd

data = pd.read_csv('train.csv')

data.head()

# %%
# filling missing values

data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

# %%
data.isnull().sum()

# %%
# convert categorical values to numbers

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

# %%
data.head()

# %%
# fix Dependents column
data['Dependents'] = data['Dependents'].replace('3+', 3)
data['Dependents'] = data['Dependents'].astype(int)

print(data['Dependents'].unique())

# %%
# split input and output

X = data.drop(columns=['Loan_ID', 'Loan_Status'])
Y = data['Loan_Status']

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X_train.shape, X_test.shape)

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# %%
from sklearn.metrics import accuracy_score

# prediction on training data
X_train_prediction = model.predict(X_train)

# calculate accuracy
accuracy = accuracy_score(Y_train, X_train_prediction)

print("Accuracy:", accuracy)

# %%
# prediction on test data
X_test_prediction = model.predict(X_test)

# calculate test accuracy
test_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Test Accuracy:", test_accuracy)

# %%
# example input data (same format as dataset)

input_data = (1, 1, 0, 1, 0, 5000, 0, 120, 360, 1, 2)

# convert to numpy array
import numpy as np
input_array = np.asarray(input_data)

# reshape the array
input_reshaped = input_array.reshape(1, -1)

# make prediction
prediction = model.predict(input_reshaped)

if prediction[0] == 1:
    print("Loan Approved ✅")
else:
    print("Loan Not Approved ❌")

# %%
import pickle

pickle.dump(model, open('model.pkl', 'wb'))

# %%



