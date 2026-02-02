import pandas as pd
import numpy as np
import nltk
import csv
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')
import re
from bs4 import BeautifulSoup
import contractions
import warnings
warnings.filterwarnings("ignore")

raw_data = pd.read_csv('./data.tsv', sep='\t', quoting=csv.QUOTE_NONE)

data = raw_data.loc[:, ['review_body', 'star_rating']]

def transform(row):
    rating = row['star_rating']
    if rating < 3:
        return 0
    elif rating > 3:
        return 1
    else:
        return 0.5

data['sentiment'] = data.apply(transform, axis=1)

print(f"Positive reviews: {len(data[data['sentiment'] == 1])}")
print(f"Negative reviews: {len(data[data['sentiment'] == 0])}")
print(f"Neutral reviews: {len(data[data['sentiment'] == 0.5])}")

positive_reviews = data[data['sentiment'] == 1].sample(100000, random_state=42)
negative_reviews = data[data['sentiment'] == 0].sample(100000, random_state=42)
data_downsized = pd.concat([positive_reviews, negative_reviews])

def clean_review(row):
    review = str(row['review_body'])
    text = review.lower()
    text = BeautifulSoup(text, "html.parser").get_text(strip=True)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = contractions.fix(text)
    return text

data_downsized['review'] = data_downsized.apply(clean_review, axis=1)

data_cleaned = data_downsized.loc[:, ['review', 'sentiment']]
print(f"Average length before cleaning: {data_downsized['review_body'].astype(str).str.len().mean():.4f}")
print(f"Average length after cleaning: {data_cleaned['review'].apply(len).mean():.4f}")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(row):
    text = row['review']
    words = word_tokenize(text)
    stop = set(stopwords.words('english'))
    return " ".join([w for w in words if w not in stop])

data_cleaned['review'] = data_cleaned.apply(remove_stopwords, axis=1)

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize(row):
    text = row['review']
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(w) for w in words])

data_cleaned['review'] = data_cleaned.apply(lemmatize, axis=1)

print(f"Average length before preprocessing: {data_downsized['review_body'].astype(str).str.len().mean():.4f}")
print(f"Average length after preprocessing: {data_cleaned['review'].apply(len).mean():.4f}")

from nltk.util import bigrams

def extract_features(row):
    text = row['review']
    tokens = word_tokenize(text)
    pairs = list(bigrams(tokens))
    return " ".join([f"{x}_{y}" for x, y in pairs])

dataset = data_cleaned.loc[:, ['sentiment']]
dataset['review'] = data_cleaned.apply(extract_features, axis=1)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

vec = TfidfVectorizer()

X = vec.fit_transform(dataset['review'])
y = dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def test_model(model, name):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    train_precision, train_recall, train_f1, train_support = precision_recall_fscore_support(y_train, y_train_pred, average="binary")
    test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(y_test, y_pred, average="binary")
    print(f"{name} Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"{name} Train Precision: {train_precision:.4f}")
    print(f"{name} Train Recall: {train_recall:.4f}")
    print(f"{name} Train F1: {train_f1:.4f}")
    print(f"{name} Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"{name} Test Precision: {test_precision:.4f}")
    print(f"{name} Test Recall: {test_recall:.4f}")
    print(f"{name} Test F1: {test_f1:.4f}")

from sklearn.linear_model import Perceptron
test_model(Perceptron(random_state=42), "Perceptron")

from sklearn.svm import LinearSVC
test_model(LinearSVC(random_state=42), "SVM")

from sklearn.linear_model import LogisticRegression
test_model(LogisticRegression(random_state=42), "Logistic Regression")

from sklearn.naive_bayes import MultinomialNB
test_model(MultinomialNB(), "Multinomial Naive Bayes")


