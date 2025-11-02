from datasets import load_dataset
import pandas as pd
import re
from pyarrow.dataset import dataset
from scipy.ndimage import vectorized_filter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


dataset = load_dataset("dair-ai/emotion")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "" , text)
    text = re.sub(r"[^a-z\s]", "" , text)
    text = re.sub(r"\s+", "", text).strip()
    return text
train_texts = [clean_text(t) for t in dataset["train"]["text"]]
train_labels = dataset["train"]["label"]



val_text = [clean_text(t) for t in dataset["validation"]["text"]]
val_labels = dataset["validation"]["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_text)
y_train = train_labels
y_val = val_labels


print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

labels = dataset["train"].features["label"].names


print(labels)


