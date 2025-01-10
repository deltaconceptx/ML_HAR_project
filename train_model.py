# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load dataset files
print("Loading datasets...")
y_train = np.loadtxt("UCI HAR Dataset/train/y_train.txt")
y_test = np.loadtxt("UCI HAR Dataset/test/y_test.txt")
subject_train = np.loadtxt("UCI HAR Dataset/train/subject_train.txt")
subject_test = np.loadtxt("UCI HAR Dataset/test/subject_test.txt")
X_train = np.loadtxt("UCI HAR Dataset/train/X_train.txt")  # Replace with actual file path for X_train.txt
X_test = np.loadtxt("UCI HAR Dataset/test/X_test.txt")    # Replace with actual file path for X_test.txt

# Print dataset shapes for verification
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Preprocess the data
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
print("Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict on the test set
print("Making predictions...")
y_pred = clf.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Generate Learning Curve
print("Generating Learning Curve...")
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train_scaled, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('results/learning_curve.png')  # Save as image
plt.show()

# Feature Importance
print("Plotting Feature Importance...")
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order

plt.figure(figsize=(10, 6))
plt.bar(range(10), importances[indices[:10]], align="center", color='skyblue')
plt.xticks(range(10), [f"Feature {i}" for i in indices[:10]], rotation=45)
plt.title("Top 10 Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.grid()
plt.savefig('results/feature_importance.png')  # Save as image
plt.show()

# Confusion Matrix
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('results/confusion_matrix.png')  # Save as image
plt.show()
# Convert confusion matrix to DataFrame
cm_df = pd.DataFrame(cm, index=np.unique(y_train), columns=np.unique(y_train))
# Save as CSV
cm_df.to_csv('results/confusion_matrix.csv')

# Save accuracy and parameters
with open('results_summary.txt', 'w') as f:
    f.write(f"Model Accuracy: {accuracy:.4f}\n")
    f.write("Random Forest Parameters:\n")
    f.write(str(clf.get_params()))
