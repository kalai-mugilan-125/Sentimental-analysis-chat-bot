from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
import nltk
import spacy
import matplotlib.pyplot as plt
import os
import random
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

app = Flask(__name__)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
# Using Qwen/Qwen2.5-7B-Instruct as the default chat model
hf_client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=hf_token) if hf_token else None


# Load and process dataset
try:
    # Changed from hardcoded 'C:/Users/Learner/Downloads/mydataset.csv' to local 'mydataset.csv'
    dataset_path = "mydataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found at {dataset_path}. Please make sure 'mydataset.csv' is in your project folder.")
    else:
        df = pd.read_csv(dataset_path, sep=";", names=["Description", "Emotion"])
        df['preprocessed_text'] = df['Description'].apply(
            lambda x: " ".join([token.lemma_ for token in nlp(x) if not token.is_stop and not token.is_punct])
        )
        label_encoder = LabelEncoder()
        df['Emotion_label'] = label_encoder.fit_transform(df['Emotion'])
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier())
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            df['preprocessed_text'], 
            df['Emotion_label'], 
            test_size=0.25, 
            random_state=42, 
            stratify=df['Emotion_label']
        )
        
        image_path = os.path.join('static', 'images', 'confusion_matrix.png')
        pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    plt.figure(figsize=(10, 7))
    cm_display.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(image_path)
    plt.close()

except Exception as e:
    print(f"Error loading dataset or training model: {e}")

# Function to get a generated response based on emotion using an LLM
def get_response(emotion, user_message):
    if not hf_client:
        return "Error: Hugging Face API token not found in .env file."
    
    messages = [
        {"role": "system", "content": "You are an empathetic conversational chatbot."},
        {"role": "user", "content": f"The user said: '{user_message}'. A simple sentiment analyzer guessed the emotion is '{emotion}', but it might be wrong. Rely on your own understanding of the user's message to respond appropriately. Reply to the user empathetically in 1 to 2 sentences, matching their true emotional context."}
    ]
    
    try:
        response = hf_client.chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            top_p=0.95
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"

# Function to classify emotion based on sentiment score
def classify_emotion(text):
    sentiment = sia.polarity_scores(text)
    compound_score = sentiment['compound']
    print(f"Sentiment Scores: {sentiment}, Compound Score: {compound_score}")
    
    if compound_score >= 0.05:
        return 'happy'     
    elif 0.03 <= compound_score < 0.05:
        return 'grateful'  
    elif 0.02 <= compound_score < 0.03:
        return 'excited'  
    elif 0.01 <= compound_score < 0.02:
        return 'hopeful'   
    elif 0 <= compound_score < 0.01:
        return 'relieved' 
    elif -0.01 <= compound_score < 0:
        return 'anxious' 
    elif -0.05 <= compound_score < -0.01:
        return 'sad'  
    elif -0.07 <= compound_score < -0.05:
        return 'disappointed'
    elif -0.1 <= compound_score < -0.07:
        return 'lonely'
    else:
        return 'angry'  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_message = request.form['msg']
    emotion = classify_emotion(user_message)
    print(f"User Message: {user_message}, Classified Emotion: {emotion}")  # Log for debugging
    bot_response = get_response(emotion, user_message)
    return jsonify({'response': bot_response})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
