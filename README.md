# Customer Churn Prediction System

A Python project that predicts whether a telecom customer is likely to churn (leave the service) using Logistic Regression, so a business can spot at-risk customers and plan retention actions.

## Dataset

 Customer Churn sample dataset :

- 7,043 customers and 21 columns
- Target: `Churn` (Yes/No)
- Saved in this repo as `data.csv`

## What the project does

1. **Load** the dataset with Pandas.
2. **Clean and validate**
   - Converted the text column `TotalCharges` to numeric
   - Found and filled 11 missing `TotalCharges` values with the median
   - Checked for duplicate rows (none found)
   - Converted `Churn` from Yes/No to 1/0
3. **Explore**
   - Overall churn rate and churn rate by contract type
   - Saves charts to `churn_distribution.png` using Matplotlib
4. **Train** a Logistic Regression model in a scikit-learn pipeline (feature scaling and one-hot encoding), with an 80/20 stratified train-test split
5. **Evaluate** with accuracy, confusion matrix, precision, recall and F1-score
6. **Predict** churn for a new customer from user input

## Results

**Exploratory analysis**

| Contract type | Churn rate |
| --- | --- |
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

Overall churn rate: 26.5%

**Model performance** (test set of 1,409 customers)

| Metric | Value |
| --- | --- |
| Accuracy | 79.56% |
| Churn precision | 0.64 |
| Churn recall | 0.54 |
| Churn F1-score | 0.58 |

Confusion matrix (rows = actual, columns = predicted):

|  | Predicted Stay | Predicted Churn |
| --- | --- | --- |
| **Actual Stay** | 919 | 116 |
| **Actual Churn** | 172 | 202 |

**Factors most linked to churn**

- Longer tenure and two-year contracts lower the chance of churn
- Customers with no internet service and those without paperless billing are less likely to churn
- Fiber optic internet and higher total charges raise the chance of churn

The model finds customers who stay easier to predict than those who leave, because the classes are imbalanced (26.5% churn). Class weighting is a possible next improvement to raise churn recall.

## Sample run

```
Enter Tenure (months): 5
Enter Monthly Charges: 70
Contract types: 1=Month-to-month, 2=One year, 3=Two year
Choose contract number: 1
Internet service: 1=DSL, 2=Fiber optic, 3=No
Choose internet service number: 2
⚠️ Customer is likely to CHURN (probability 71.4%)
```

## Technologies

Python, Pandas, Scikit-learn, Matplotlib

## How to run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the project:

```
python main.py
```

## Project files

- `main.py`: data cleaning, analysis, model training, evaluation and prediction
- `data.csv`: Churn dataset
- `requirements.txt`: Python libraries needed
- `churn_distribution.png`: churn distribution and churn by charts


