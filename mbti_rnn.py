import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ===== TensorFlow/Keras Import =====
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf

    print("Using TensorFlow v{}".format(tf.__version__))
except ImportError as e:
    raise ImportError("Required TensorFlow/Keras libraries not found. Please install with: pip install tensorflow")

# ===== GPU Acceleration Configuration =====
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s) for acceleration")
    except RuntimeError as e:
        print(e)


# ================== Confusion Matrix Visualization Function ==================
def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    """Visualize confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, cbar=False)

    plt.title(title, pad=20, fontsize=14)
    plt.xlabel('Predicted Label', labelpad=15)
    plt.ylabel('True Label', labelpad=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# ================== Data Loading and Preprocessing ==================
print("Loading and preprocessing data...")
data = pd.read_csv('mbti_1.csv')
data['cleaned_posts'] = data['posts'].apply(lambda x: ' '.join(
    re.sub(r"http\S+|[^a-zA-Z\s]", "", x).lower().split()))

# Label encoding
le = LabelEncoder()
y = le.fit_transform(data['type'])

# Split dataset
X_train_text, X_val_text, y_train, y_val = train_test_split(
    data['cleaned_posts'], y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ================== Optimized LSTM Model ==================
def train_lstm():
    print("\n" + "=" * 50)
    print("Training Optimized LSTM Model")
    print("=" * 50)

    # Text sequence processing
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(X_train_text)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=100)
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val_text), maxlen=100)

    # Model configuration
    model = Sequential([
        Embedding(2000, 64),  # Reduced embedding dimension for faster training
        Dropout(0.3),
        LSTM(64, dropout=0.2),  # Reduced units
        Dense(16, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipvalue=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training loop
    lstm_accuracies = []
    for i in tqdm(range(5), desc="LSTM Training"):
        partial_X_train, _, partial_y_train, _ = train_test_split(
            X_train_seq, y_train,
            test_size=0.2,
            random_state=i,
            stratify=y_train
        )

        model.fit(
            partial_X_train, partial_y_train,
            epochs=15,
            batch_size=256,  # Increased batch size for faster training
            verbose=0,
            callbacks=[EarlyStopping(patience=2)]
        )

        y_pred = np.argmax(model.predict(X_val_seq, verbose=0), axis=1)
        lstm_accuracies.append(accuracy_score(y_val, y_pred))

    return tokenizer, model, lstm_accuracies


tokenizer, lstm_model, lstm_accuracies = train_lstm()

# ================== Traditional Model Training ==================
print("\n" + "=" * 50)
print("Training Traditional Models")
print("=" * 50)

# TF-IDF features
vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.1, max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf = vectorizer.transform(X_val_text)

# Model definitions
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.01, random_state=42),
    "SVM": SVC(kernel='linear', C=1, probability=True)
}

# Training process and prediction storage
model_accuracies = {"LSTM": lstm_accuracies}
model_predictions = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    accuracies = []
    full_model = model.__class__(**model.get_params())  # Create new instance for full training

    # Full training for confusion matrix
    full_model.fit(X_train_tfidf, y_train)
    model_predictions[name] = full_model.predict(X_val_tfidf)

    # Cross-validation accuracy
    for i in range(5):
        partial_X_train, _, partial_y_train, _ = train_test_split(
            X_train_tfidf, y_train,
            test_size=0.2,
            random_state=i,
            stratify=y_train
        )
        model.fit(partial_X_train, partial_y_train)
        y_pred = model.predict(X_val_tfidf)
        accuracies.append(accuracy_score(y_val, y_pred))
    model_accuracies[name] = accuracies

# ================== Result Visualization ==================
print("\n" + "=" * 50)
print("Generating Visualizations")
print("=" * 50)

# Results table
results = {
    "LSTM": np.mean(lstm_accuracies),
    **{name: np.mean(acc) for name, acc in model_accuracies.items() if name != "LSTM"}
}
results_df = pd.DataFrame(list(results.items()), columns=["Algorithm", "Validation Accuracy"])
results_df = results_df.sort_values("Validation Accuracy", ascending=False)

# 1. Results table plot
plt.figure(figsize=(10, 4))
ax = plt.subplot(111)
ax.axis('off')
plt.title("Model Performance Comparison", pad=20)
table = ax.table(
    cellText=results_df.round(4).values,
    colLabels=results_df.columns,
    cellLoc="center",
    loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.savefig("results_table.png", bbox_inches='tight')

# 2. Accuracy comparison line plot
plt.figure(figsize=(10, 6))
x = range(1, 6)
for name, acc in model_accuracies.items():
    plt.plot(x, acc, marker='o', label=name, linewidth=2)
plt.title("Validation Accuracy Across Iterations", pad=15)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.xticks(x)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)

# 3. Confusion matrix visualization
print("\nGenerating Confusion Matrices...")

# LSTM confusion matrix
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val_text), maxlen=100)
y_pred_lstm = np.argmax(lstm_model.predict(X_val_seq, verbose=0), axis=1)
plot_confusion_matrix(y_val, y_pred_lstm, le.classes_,
                      "LSTM Confusion Matrix", "lstm_confusion.png")

# Traditional model confusion matrices
for name, y_pred in model_predictions.items():
    plot_confusion_matrix(y_val, y_pred, le.classes_,
                          f"{name} Confusion Matrix",
                          f"{name.lower().replace(' ', '_')}_confusion.png")

print("\nFinal Results Table:")
print(results_df)
print("\nVisualizations saved as:")
print("- results_table.png")
print("- accuracy_comparison.png")
print("- lstm_confusion.png")
print("- random_forest_confusion.png")
print("- xgboost_confusion.png")
print("- svm_confusion.png")
