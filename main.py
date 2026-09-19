"""Customer Churn Prediction System (IBM Telco Customer Churn dataset)

Steps: load data -> clean -> explore -> train -> evaluate -> predict.
Run with:  python main.py
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = ["Contract", "InternetService",
               "PaymentMethod", "PaperlessBilling"]
TARGET = "Churn"
REQUIRED = set(NUMERIC + CATEGORICAL + [TARGET])

# ---------------------------------------------------------------
# 1. Load data (finds the Telco CSV in this folder automatically)
# ---------------------------------------------------------------
data = None
for path in ["data.csv"] + sorted(glob.glob("*.csv")):
    try:
        df = pd.read_csv(path)
    except Exception:
        continue
    df.columns = df.columns.str.strip()
    if REQUIRED.issubset(df.columns):
        data, used_file = df, path
        break

if data is None:
    raise SystemExit(
        "Could not find the Telco churn CSV in this folder. "
        "Put the file here (it needs columns like tenure, MonthlyCharges, "
        "TotalCharges, Contract and Churn)."
    )

print("Customer Churn Prediction System")
print("--------------------------------")
print(f"Loaded file: {used_file}")
print(f"Dataset size: {data.shape[0]} rows, {data.shape[1]} columns")

# ---------------------------------------------------------------
# 2. Data cleaning and validation
# ---------------------------------------------------------------
# TotalCharges is stored as text and can contain blank values
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
for col in NUMERIC:
    data[col] = pd.to_numeric(data[col], errors="coerce")

print("\nMissing values before cleaning:")
missing = data[NUMERIC + CATEGORICAL + [TARGET]].isnull().sum()
print(missing[missing > 0] if missing.sum() else "None")

# Churn: Yes/No -> 1/0
data[TARGET] = data[TARGET].astype(
    str).str.strip().str.lower().map({"yes": 1, "no": 0})
rows_before = len(data)
data = data.dropna(subset=[TARGET])
print(
    f"\nRows removed (missing/invalid churn label): {rows_before - len(data)}")

# Fill missing numeric values with the median
for col in NUMERIC:
    n_missing = data[col].isnull().sum()
    if n_missing:
        data[col] = data[col].fillna(data[col].median())
        print(f"Filled {n_missing} missing values in '{col}' with the median")

# Tidy category text and fill any blanks
for col in CATEGORICAL:
    data[col] = data[col].astype(str).str.strip()

data[TARGET] = data[TARGET].astype(int)
print(f"Duplicate rows found: {data.duplicated().sum()}")
print(f"Rows after cleaning: {len(data)}")

# ---------------------------------------------------------------
# 3. Exploratory data analysis
# ---------------------------------------------------------------
churn_rate = data[TARGET].mean() * 100
print(f"\nChurn rate: {churn_rate:.1f}%")
print("\nChurn rate by contract type (%):")
by_contract = (data.groupby("Contract")[TARGET].mean() * 100).round(1)
print(by_contract)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
counts = data[TARGET].value_counts().sort_index()
axes[0].bar(["Stay", "Churn"], [counts.get(0, 0), counts.get(1, 0)],
            color=["#4c9f70", "#d9534f"])
axes[0].set_title("Customer Churn Distribution")
axes[0].set_ylabel("Number of customers")
axes[1].bar(by_contract.index, by_contract.values, color="#5b8def")
axes[1].set_title("Churn Rate by Contract Type")
axes[1].set_ylabel("Churn rate (%)")
axes[1].tick_params(axis="x", labelrotation=15)
plt.tight_layout()
plt.savefig("churn_distribution.png", dpi=150)
plt.close()
print("Saved chart: churn_distribution.png")

# ---------------------------------------------------------------
# 4. Train the model
# ---------------------------------------------------------------
X = data[NUMERIC + CATEGORICAL]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

# ---------------------------------------------------------------
# 5. Evaluate the model
# ---------------------------------------------------------------
y_pred = model.predict(X_test)

print("\nModel evaluation (test set)")
print("---------------------------")
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report (precision / recall / f1):")
print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

# Top factors linked to churn (positive = more likely to churn)
names = model.named_steps["prep"].get_feature_names_out()
coefs = model.named_steps["clf"].coef_[0]
top = sorted(zip(names, coefs), key=lambda p: abs(p[1]), reverse=True)[:6]
print("Top factors influencing churn (scaled coefficients):")
for name, value in top:
    print(f"  {name.split('__', 1)[1]}: {value:+.3f}")

# ---------------------------------------------------------------
# 6. Predict for a new customer
# ---------------------------------------------------------------
print("\nPredict churn for a new customer")
print("--------------------------------")
contracts = sorted(data["Contract"].unique())
internet = sorted(data["InternetService"].unique())

try:
    tenure = int(input("Enter Tenure (months): "))
    monthly = float(input("Enter Monthly Charges: "))
    print("Contract types:", ", ".join(
        f"{i + 1}={c}" for i, c in enumerate(contracts)))
    contract = contracts[int(input("Choose contract number: ")) - 1]
    print("Internet service:", ", ".join(
        f"{i + 1}={c}" for i, c in enumerate(internet)))
    service = internet[int(input("Choose internet service number: ")) - 1]
except (ValueError, IndexError):
    raise SystemExit("Invalid input. Please enter valid numbers.")

# Other fields use typical values from the dataset
new_customer = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": tenure * monthly,
    "SeniorCitizen": 0,
    "Contract": contract,
    "InternetService": service,
    "PaymentMethod": data["PaymentMethod"].mode()[0],
    "PaperlessBilling": data["PaperlessBilling"].mode()[0],
}])

prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0][1] * 100

if prediction == 1:
    print(f"⚠️ Customer is likely to CHURN (probability {probability:.1f}%)")
else:
    print(
        f"✅ Customer is likely to STAY (churn probability {probability:.1f}%)")

