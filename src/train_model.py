import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from collections import Counter
import joblib

data = []
labels = []

for file in os.listdir("dataset"):
    gesture = file.split(".")[0]
    df = pd.read_csv(f"dataset/{file}", header=None)
    for row in df.values:
        data.append(row)
        labels.append(gesture)


print("Checking dataset balance...")
print(Counter(labels))

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
# Basic metric
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
# Advanced metrics
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Optional: individual metrics
print("\nPrecision:", precision_score(y_test, predictions, average='weighted'))
print("Recall:", recall_score(y_test, predictions, average='weighted'))
print("F1-score:", f1_score(y_test, predictions, average='weighted'))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/gesture_model.pkl")
