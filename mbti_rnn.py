import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

# Load dataset
data = pd.read_csv('mbti_1.csv')
data['cleaned_posts'] = data['posts'].apply(lambda x: ' '.join(
    re.sub(r"http\S+|[^a-zA-Z\s]", "", x).lower().split()))

# Feature engineering: TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
print("Processing TF-IDF features...")
vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.1, max_features=2000)  # Reduce feature size to save memory
X = vectorizer.fit_transform(tqdm(data['cleaned_posts'], desc="Vectorizing posts"))

# Encode labels
from sklearn.preprocessing import LabelEncoder
print("Encoding labels...")
y = LabelEncoder().fit_transform(data['type'])

# Split dataset
print("Splitting dataset...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Models to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.01, random_state=42),
    "SVM": SVC(kernel='linear', C=1, probability=True)
}

# Train and validate models
results = {}
validation_accuracies = {name: [] for name in models.keys()}  # Track accuracy over iterations
print("Training and validating models...")
for name, model in tqdm(models.items(), desc="Evaluating models"):
    for i in range(1, 6):  # Simulate incremental training (5 steps)
        partial_X_train, partial_X_val, partial_y_train, partial_y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=i, stratify=y_train)
        model.fit(partial_X_train, partial_y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        validation_accuracies[name].append(accuracy)
    results[name] = validation_accuracies[name][-1]

# Convert results to DataFrame
results_df = pd.DataFrame(list(results.items()), columns=["Algorithm", "Validation Accuracy"])
results_df.sort_values(by="Validation Accuracy", ascending=False, inplace=True)

# Plot results as a table
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(results_df.columns))))
plt.savefig("algorithm_comparison_table.png", bbox_inches='tight')

# Plot results as a line chart
plt.figure(figsize=(8, 5))
plt.plot(results_df["Algorithm"], results_df["Validation Accuracy"], marker='o', linestyle='-', color='b')
plt.title("Algorithm Validation Accuracy Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.savefig("algorithm_accuracy_linechart.png", bbox_inches='tight')

# Plot validation accuracy over iterations
plt.figure(figsize=(10, 6))
for name, accuracies in validation_accuracies.items():
    plt.plot(range(1, 6), accuracies, marker='o', label=name)
plt.title("Validation Accuracy Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("validation_accuracy_iterations.png", bbox_inches='tight')

# Output results
print("Validation Accuracy of Models:")
print(results_df)
