import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import warnings
from transformers import EarlyStoppingCallback

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

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

Label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
results = {}

def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True)

for label in Label_columns:
    print(f"\nTraining model for {label}...\n")

    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_comment_text'], y, test_size=0.2, random_state=42, stratify=y
    )

    # Tokenize data
    train_encodings = tokenizer(list(X_train), padding=True, truncation=True, max_length=512)
    test_encodings = tokenizer(list(X_test), padding=True, truncation=True, max_length=512)

    class ToxicDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_dataset = ToxicDataset(train_encodings, y_train.values)
    test_dataset = ToxicDataset(test_encodings, y_test.values)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Define training arguments with early stopping
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory
        evaluation_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",  # Save every epoch
        learning_rate=5e-5,  # Learning rate
        per_device_train_batch_size=8,  # Batch size for training
        per_device_eval_batch_size=16,  # Batch size for evaluation
        num_train_epochs=10,  # Set a higher number of epochs, early stopping will handle the stopping
        weight_decay=0.01,  # Weight decay
        logging_dir="./logs",  # Directory for logs
        logging_steps=10,
        load_best_model_at_end=True,  # Load the best model based on evaluation metrics
        metric_for_best_model="accuracy",  # Metric to track (can use "accuracy", "eval_loss", etc.)
    )

    # Define EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2)  # Stop after 2 evaluation steps with no improvement

    # Define Trainer with the early stopping callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[early_stopping_callback],  # Add early stopping callback here
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(f"./saved_model_{label}")
    tokenizer.save_pretrained(f"./saved_model_{label}")

    # Evaluate the model
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    report = classification_report(y_test, preds, output_dict=True)
    results[label] = report

    print(f"Classification Report for {label}:")
    print(classification_report(y_test, preds))

print("\nSummary of Precision Scores:")
for label, report in results.items():
    print(f"{label}: Precision = {report['1']['precision']:.4f}")

results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv('classification_reports_pretrained.csv', index=True)
