import pandas as pd

# Create student dataset

data = {
    "Study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [20, 30, 40, 50, 60, 70, 80, 90]
}

# convert data into dataframe

df = pd.DataFrame(data)

# display dataset

print(df)

# display dataset information

print("\nDataset Information:")
print(df.info())

# display statistical information

print("\nDataset Statistics:")
print(df.describe())



#### Create graph ############################## matplotlib - used to create graph and charts 

import matplotlib.pyplot as plt

plt.scatter(df['Study_hours'], df['Marks'])

# Add Labels and title

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Study Hours vs Marks")

# display graph
# plt.show()
plt.savefig("student_marks_graph.png")

print("Graph saved successfully!")



# 2. Understand Features (X) and Target (y)
# In Machine Learning:
# - Feature (X): Input data used for prediction.    "x contains study hours"
# - Target (y): Output we want to predict.           "y contains marks"

# 3. Split Training and Testing Data
# Training data – Used to teach the model.
# Testing data – Used to evaluate the model on unseen records.

from sklearn.model_selection import train_test_split

# Extract Features (X) and Target (y)

X = df[['Study_hours']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:")
print(X_train)

print("Testing Data:")
print(X_test)

# 4. Train Your First ML Model

# | Operation | Code | Purpose |
# |---|---|---|
# | Training | `model.fit()` | Model learns from data |
# | Prediction | `model.predict()` | Predicts new values |
# | Data splitting | `train_test_split()` | Separates training and testing data |

from sklearn.linear_model import LinearRegression

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

# 5. Make Your First Prediction
# Now let's predict marks for a student who studies for 9 hours.

prediction = model.predict([[4]])

print("Predicted Marks:", prediction[0])