import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("data.csv")

# Features and target
X = data[['age', 'monthly_charges', 'tenure']]
y = data['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Customer Churn Prediction System")
print("--------------------------------")

# User input
age = int(input("Enter Age: "))
monthly_charges = float(input("Enter Monthly Charges: "))
tenure = int(input("Enter Tenure (months): "))

# Predict
prediction = model.predict([[age, monthly_charges, tenure]])

if prediction[0] == 1:
    print("⚠️ Customer is likely to CHURN")
else:
    print("✅ Customer is likely to STAY")
