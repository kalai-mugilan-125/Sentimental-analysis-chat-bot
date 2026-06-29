# Emotion-Based Chatbot using Flask, NLTK, and Machine Learning

## Overview

This project is an interactive chatbot built using Flask that can classify the user's emotional sentiment based on their text input and generate empathetic responses. The chatbot leverages NLTK's SentimentIntensityAnalyzer, Spacy, and a machine learning pipeline built with RandomForestClassifier for emotion classification. Instead of hardcoded replies, it uses the **Hugging Face Serverless Inference API (Qwen model)** to dynamically generate contextual, empathetic responses based on the user's message and detected emotion.

## Features

- Sentiment Analysis using NLTK's VADER lexicon.
- Text preprocessing and lemmatization using Spacy.
- Emotion classification with a RandomForest Classifier model.
- Dynamic AI-generated chatbot responses using Hugging Face (Qwen2.5-7B-Instruct model) based on the detected emotion.
- Context-aware AI that can correct misclassifications from the simple sentiment analyzer by deeply understanding the user's message.
- A simple user interface using HTML/CSS and Bootstrap.
- Displays a confusion matrix to visualize the model's performance.

## Project Structure

project_root/
│
├── static/
│   ├── css/
│   ├── images/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── .env                 # Environment variables for API Keys
├── app.py               # Main Flask application logic
├── mydataset.csv        # Dataset used for training the emotion model
├── requirements.txt     # Python dependencies
└── README.md            

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/kalai-mugilan-125/Sentimental-analysis-chat-bot
   cd project_root
   ```

2. **Set up API Keys**
   Create a `.env` file in the root directory (or use the provided template) and add your Hugging Face access token:
   ```env
   HUGGINGFACE_API_TOKEN=your_huggingface_token_here
   ```
   You can generate a free token at [Hugging Face Settings](https://huggingface.co/settings/tokens).

3. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK's VADER lexicon and Spacy's language model**
   ```bash
   python -m nltk.downloader vader_lexicon
   python -m spacy download en_core_web_sm
   ```

5. **Run the Flask application**
   ```bash
   python app.py
   ```
   Open your browser and visit `http://localhost:5001` to interact with the chatbot.

6. **Dataset**
   The chatbot is trained using a CSV file (`mydataset.csv`) containing user descriptions and corresponding emotions. The dataset is preprocessed using Spacy to remove stop words and punctuation, followed by text lemmatization.

## Response Logic

The chatbot uses a combination of Sentiment Analysis and Emotion Classification to determine the user's emotional state. The following emotions are detected based on the text's sentiment score:

- **Positive Emotions:** Happy, Grateful, Excited
- **Neutral Emotions:** Hopeful, Relieved
- **Negative Emotions:** Angry, Sad, Lonely, etc.

Once the emotion is roughly classified, both the emotion and the user's original message are sent to a **Qwen2.5-7B LLM** via the Hugging Face API. The LLM is instructed to rely on its own deep understanding of the user's text to generate a responsive, contextually aware, and empathetic reply, effectively overcoming any potential misclassifications by the simple sentiment analyzer.

## Dependencies

- Flask
- NLTK
- Spacy
- scikit-learn
- Pandas
- Matplotlib
- huggingface-hub
- python-dotenv

You can install all dependencies easily via:
```bash
pip install -r requirements.txt
```
