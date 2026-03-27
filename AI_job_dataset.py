import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("ai_job_dataset.csv")
df = pd.DataFrame(data)

X = df[['years_experience']]   
y = df['salary_usd']           

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

m = model.coef_[0]
c = model.intercept_

print(f"Linear Regression Equation: Salary = {m:.2f} × Years of Experience + {c:.2f}")

user_exp = float(input("Enter Years of Experience: "))
user_exp_df = pd.DataFrame({'years_experience': [user_exp]})

predicted_salary = model.predict(user_exp_df)

print(f"Predicted Salary: ${predicted_salary[0]:.2f}")