import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import Counter
# Function to create individual feature plots
import matplotlib.pyplot as plt
import numpy as np


nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



def clean_text(text):
    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    words = re.sub(r'[^\w\s]', '', text).split()
    words = [word.lower() for word in words if word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words



# Load the CSV file
file_path = 'train.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Load the CSV file
file_path = 'profanity_en.csv'  # Replace with your file path
profanity_data = pd.read_csv(file_path)

# Combine all canonical forms into a single list
profanity_list = (
    pd.concat([
        profanity_data['canonical_form_1'].dropna(),
        profanity_data['canonical_form_2'].dropna(),
        profanity_data['canonical_form_3'].dropna()
    ])
    .drop_duplicates()  # Remove duplicates
    .tolist()           # Convert to a Python list
)

# Initialize the data structures to hold the feature values for each label
toxic = {
    'length': [],
    'word_count': [],
    'capital_letters': [],
    'exclamation_marks': [],
    'question_marks': [],
    'profanity_count': [],
    'special_characters': [],
    'sentence_count': [],
    'average_word_length': [],
    'stopword_ratio': [],
    'unique_word_ratio': [],
    'word_repetition': [],
    'sentiment_polarity': [],
    'sentiment_subjectivity': []
}

severe = {
    'length': [],
    'word_count': [],
    'capital_letters': [],
    'exclamation_marks': [],
    'question_marks': [],
    'profanity_count': [],
    'special_characters': [],
    'sentence_count': [],
    'average_word_length': [],
    'stopword_ratio': [],
    'unique_word_ratio': [],
    'word_repetition': [],
    'sentiment_polarity': [],
    'sentiment_subjectivity': []
}

obscene = {
    'length': [],
    'word_count': [],
    'capital_letters': [],
    'exclamation_marks': [],
    'question_marks': [],
    'profanity_count': [],
    'special_characters': [],
    'sentence_count': [],
    'average_word_length': [],
    'stopword_ratio': [],
    'unique_word_ratio': [],
    'word_repetition': [],
    'sentiment_polarity': [],
    'sentiment_subjectivity': []
}

threat = {
    'length': [],
    'word_count': [],
    'capital_letters': [],
    'exclamation_marks': [],
    'question_marks': [],
    'profanity_count': [],
    'special_characters': [],
    'sentence_count': [],
    'average_word_length': [],
    'stopword_ratio': [],
    'unique_word_ratio': [],
    'word_repetition': [],
    'sentiment_polarity': [],
    'sentiment_subjectivity': []
}

insult = {
    'length': [],
    'word_count': [],
    'capital_letters': [],
    'exclamation_marks': [],
    'question_marks': [],
    'profanity_count': [],
    'special_characters': [],
    'sentence_count': [],
    'average_word_length': [],
    'stopword_ratio': [],
    'unique_word_ratio': [],
    'word_repetition': [],
    'sentiment_polarity': [],
    'sentiment_subjectivity': []
}

identity = {
    'length': [],
    'word_count': [],
    'capital_letters': [],
    'exclamation_marks': [],
    'question_marks': [],
    'profanity_count': [],
    'special_characters': [],
    'sentence_count': [],
    'average_word_length': [],
    'stopword_ratio': [],
    'unique_word_ratio': [],
    'word_repetition': [],
    'sentiment_polarity': [],
    'sentiment_subjectivity': []
}


# Function to analyze each row
def analyze_row(row, profanity_list, toxic, severe, obscene, threat, insult, identity):
    comment_text = row['comment_text']
    toxicity_labels = row[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    # Cleaning and tokenizing text
    words = clean_text(comment_text)

    # Calculate metrics
    features = {
        'length': len(comment_text),
        'word_count': len(words),
        'capital_letters': sum(1 for c in comment_text if c.isupper()),
        'exclamation_marks': comment_text.count('!'),
        'question_marks': comment_text.count('?'),
        'profanity_count': sum(word in profanity_list for word in words),
        'special_characters': sum(1 for c in comment_text if not c.isalnum() and not c.isspace()),
        'sentence_count': len(re.split(r'[.!?]', comment_text)) - 1,
        'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'stopword_ratio': sum(1 for word in words if word in stop_words) / len(words) if words else 0,
        'unique_word_ratio': len(set(words)) / len(words) if words else 0,
        'word_repetition': sum(count > 1 for count in Counter(words).values()),
        'sentiment_polarity': TextBlob(comment_text).sentiment.polarity,
        'sentiment_subjectivity': TextBlob(comment_text).sentiment.subjectivity,
    }

    # Populate the appropriate label feature dictionaries
    if toxicity_labels['toxic']:
        for key in toxic:
            toxic[key].append(features[key])

    if toxicity_labels['severe_toxic']:
        for key in severe:
            severe[key].append(features[key])

    if toxicity_labels['obscene']:
        for key in obscene:
            obscene[key].append(features[key])

    if toxicity_labels['threat']:
        for key in threat:
            threat[key].append(features[key])

    if toxicity_labels['insult']:
        for key in insult:
            insult[key].append(features[key])

    if toxicity_labels['identity_hate']:
        for key in identity:
            identity[key].append(features[key])

    return features

df.apply(
    analyze_row,
    axis=1,
    args=(profanity_list, toxic, severe, obscene, threat, insult, identity)
)



def plot_feature_avg(feature_name, toxic, severe, obscene, threat, insult, identity, unit=''):
    # Prepare the data for plotting
    labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity']
    feature_values = [
        np.mean(toxic[feature_name]),
        np.mean(severe[feature_name]),
        np.mean(obscene[feature_name]),
        np.mean(threat[feature_name]),
        np.mean(insult[feature_name]),
        np.mean(identity[feature_name])
    ]

    # Define colors for each label
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

    # Plot the data
    plt.figure(figsize=(15, 5))
    bars = plt.bar(labels, feature_values, color=colors)

    # Add legend
    plt.legend(bars, labels, title="Labels", loc='upper right')

    # Adding titles and labels
    plt.title(f'Average {feature_name.capitalize()} for Different Labels', fontsize=16)
    plt.xlabel('Toxicity Label', fontsize=12)
    plt.ylabel(f'Average {feature_name.capitalize()} ({unit})', fontsize=12)

    # Show the plot
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'output_{feature_name}.png', dpi = 700)

    # Save averages to a text file
    with open(f'{feature_name}_averages.txt', 'w') as file:
        file.write(f'Feature: {feature_name.capitalize()} ({unit})\n\n')
        for label, value in zip(labels, feature_values):
            file.write(f'{label}: {value:.4f}\n')

    print(f"Averages saved to {feature_name}_averages.txt")


# Plot features and save their averages
plot_feature_avg('length', toxic, severe, obscene, threat, insult, identity, unit='characters')
plot_feature_avg('word_count', toxic, severe, obscene, threat, insult, identity, unit='words')
plot_feature_avg('capital_letters', toxic, severe, obscene, threat, insult, identity, unit='letters')
plot_feature_avg('exclamation_marks', toxic, severe, obscene, threat, insult, identity, unit='marks')
plot_feature_avg('question_marks', toxic, severe, obscene, threat, insult, identity, unit='marks')
plot_feature_avg('profanity_count', toxic, severe, obscene, threat, insult, identity, unit='words')
plot_feature_avg('special_characters', toxic, severe, obscene, threat, insult, identity, unit='characters')
plot_feature_avg('sentence_count', toxic, severe, obscene, threat, insult, identity, unit='sentences')
plot_feature_avg('average_word_length', toxic, severe, obscene, threat, insult, identity, unit='characters')
plot_feature_avg('stopword_ratio', toxic, severe, obscene, threat, insult, identity, unit='ratio')
plot_feature_avg('unique_word_ratio', toxic, severe, obscene, threat, insult, identity, unit='ratio')
plot_feature_avg('word_repetition', toxic, severe, obscene, threat, insult, identity, unit='count')
plot_feature_avg('sentiment_polarity', toxic, severe, obscene, threat, insult, identity, unit='score')
plot_feature_avg('sentiment_subjectivity', toxic, severe, obscene, threat, insult, identity, unit='score')
