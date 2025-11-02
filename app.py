import streamlit as st
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
import joblib

# Load dataset and labels
dataset = load_dataset("dair-ai/emotion")
labels = dataset["train"].features["label"].names

# Define cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prepare and train quickly (you can later load a saved model)
train_texts = [clean_text(t) for t in dataset["train"]["text"]]
train_labels = dataset["train"]["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# Prediction function
def predict_emotion(text):
    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    pred = model.predict(vector)[0]
    emotion = labels[pred]
    return emotion

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Detector", page_icon="🕶️")

st.title("🕶️ Emotion Detection from Text")
st.write("Type a sentence below and let the AI guess your emotion:")

user_input = st.text_area("Enter text:", "")

if st.button("Predict Emotion"):
    if user_input.strip():
        emotion = predict_emotion(user_input)
        st.success(f"**Predicted Emotion:** {emotion.capitalize()}")
    else:
        st.warning("Please type something first!")
