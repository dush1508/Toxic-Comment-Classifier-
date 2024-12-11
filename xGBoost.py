import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)

# Load data
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' file not found.")
    exit()

# Define stopwords
stop_words = set(stopwords.words('english'))


def clean_text_v2(text):
    if pd.isnull(text):
        return '<empty>'
    text = re.sub(r'[^\w\s!@#$%^&*]', '', text.lower())
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


# Apply text cleaning
df['cleaned_comment_text'] = df['comment_text'].apply(clean_text_v2)

# Encode cleaned text into embeddings
model = SentenceTransformer("all-mpnet-base-v2")
X = model.encode(df['cleaned_comment_text'].tolist(), batch_size=32, show_progress_bar=True)

Label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
results = {}

# Define parameter grid for manual search
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 5, 10],
}

for label in Label_columns:
    print(f"\nTraining model for {label}...\n")

    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE only on training data
    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    best_f1_score = -np.inf
    best_model = None

    # Manually loop through all parameter combinations
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for n_estimators in param_grid['n_estimators']:
                for min_child_weight in param_grid['min_child_weight']:

                    model = xgb.XGBClassifier(
                        objective='binary:logistic',
                        random_state=42,
                        eval_metric='logloss',
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        min_child_weight=min_child_weight
                    )

                    model.fit(X_train_resampled, y_train_resampled)
                    y_pred = model.predict(X_test)

                    # Calculate classification report and F1 score
                    report = classification_report(y_test, y_pred, output_dict=True)
                    f1_score = report['1']['f1-score']

                    # Update best model if F1 score is better
                    if f1_score > best_f1_score:
                        best_f1_score = f1_score
                        best_model = model

    # Get predictions from the best model and generate the classification report
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[label] = report

    print(f"Classification Report for {label}:")
    print(classification_report(y_test, y_pred))

print("\nSummary of Precision Scores:")
for label, report in results.items():
    print(f"{label}: Precision = {report['1']['precision']:.4f}")

results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv('classification_reports.csv', index=True)
