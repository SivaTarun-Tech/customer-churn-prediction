from model import train_model
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data.csv")

# Train model
model = train_model()

# Show insights
def show_insights():
    print("\n--- DATA INSIGHTS ---")
    print("Total Customers:", len(data))
    print("Churn Rate: {:.2f}%".format(data['churn'].mean() * 100))

# Show graph
def show_graph():
    churn_counts = data['churn'].value_counts()

    plt.figure()
    plt.bar(['Stay', 'Churn'], churn_counts)
    plt.title("Customer Churn Distribution")
    plt.xlabel("Customer Status")
    plt.ylabel("Count")
    plt.show()

# Predict churn
def predict():
    print("\nEnter Customer Details:")

    age = int(input("Age: "))
    monthly_charges = float(input("Monthly Charges: "))
    tenure = int(input("Tenure (months): "))

    prediction = model.predict([[age, monthly_charges, tenure]])

    if prediction[0] == 1:
        print("⚠️ Customer is likely to CHURN")
    else:
        print("✅ Customer is likely to STAY")

# Menu system
def menu():
    while True:
        print("\n1. Show Insights")
        print("2. Predict Churn")
        print("3. Show Graph")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            show_insights()
        elif choice == "2":
            predict()
        elif choice == "3":
            show_graph()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice")

# Run program
if __name__ == "__main__":
    menu()
