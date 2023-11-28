#Ahnaf Ahmad
#1001835014

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('glass.txt', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=32)

rf.fit(X_train, y_train)

train_predictions = rf.predict(X_train)
test_predictions = rf.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Machine Learning Algorithm: Random Forest")
print("Library: SK Learn")
print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Testing Accuracy: {test_accuracy*100:.2f}%")

test = [[1.51761, 12.81, 3.54, 1.23, 73.24, 0.58, 8.39, 0.00, 0.00],
          [1.51514, 14.01, 2.68, 3.50, 69.89, 1.68, 5.87, 2.20, 0.00],
          [1.51937, 13.79, 2.41, 1.19, 72.76, 0.00, 9.77, 0.00, 0.00],
          [1.51658, 14.80, 0.00, 1.99, 73.11, 0.00, 8.28, 1.71, 0.00]]

classes = ["building windows", "housing windows", "vehicle windows", "trucking windows", "containers", "tableware", "headlamps"]

predictions = rf.predict(test)

print("Predicted Classes:")
for i, prediction in enumerate(predictions):
    print(f"Input {i+1}: {classes[prediction-1]}")

