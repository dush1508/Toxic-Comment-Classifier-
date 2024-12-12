import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
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

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_comment_text'])
X = tokenizer.texts_to_sequences(df['cleaned_comment_text'])
X = pad_sequences(X, maxlen=100)  # Padding sequences to a maximum length of 100

Label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
results = {}

# Define LSTM model architecture
def create_lstm_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=100),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

vocab_size = len(tokenizer.word_index) + 1  # Size of vocabulary

for label in Label_columns:
    print(f"\nTraining model for {label}...\n")

    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE only on training data
    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    # Create and train the LSTM model
    model = create_lstm_model(vocab_size)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(
        X_train_resampled, y_train_resampled,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    # Save the trained model
    model.save(f"lstm_model_{label}.h5")

    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    report = classification_report(y_test, y_pred, output_dict=True)
    results[label] = report

    print(f"Classification Report for {label}:")
    print(classification_report(y_test, y_pred))

print("\nSummary of Precision Scores:")
for label, report in results.items():
    print(f"{label}: Precision = {report['1']['precision']:.4f}")

results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv('classification_reports.csv', index=True)
