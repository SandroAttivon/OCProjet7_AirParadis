from flask import Flask, request, jsonify
import pandas as pd
import joblib
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import os
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

app = Flask(__name__)

# Application Insights logger setup
APPINSIGHTS_CONNECTION_STRING = os.environ.get("APPINSIGHTS_CONNECTION_STRING", "")
logger = logging.getLogger(__name__)
if APPINSIGHTS_CONNECTION_STRING:
    logger.addHandler(AzureLogHandler(connection_string=APPINSIGHTS_CONNECTION_STRING))


# port = int(os.environ.get("PORT", 5500))  # Azure injecte PORT (ex: 8000 ou 80)
# app.run(host="0.0.0.0", port=port)

# 📂 Chemins vers les modèles et données
MODEL_PATH = "./models/model_logreg.pkl"
VECTORIZER_PATH = "./models/tfidf_vectorizer.pkl"

# 🔁 Chargement du modèle et du vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# 🧹 Fonction de nettoyage
# def preprocess(text):
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r"@\w+|#", '', text)
#     text = re.sub(r"[^\w\s]", '', text.lower())
#     return text

# 🧹 Advanced cleaning process (NLTK-free stopwords, tokenizer, stemmer, lemmatizer)
STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
             'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
             'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
             'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
             'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
             'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
             'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
             'with', 'about', 'against', 'between', 'into', 'through', 'during',
             'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
             'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text, method='lemma'):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#", '', text)
    text = re.sub(r"[^\w\s]", '', text.lower())
    words = simple_tokenize(text)
    words = [w for w in words if w not in STOPWORDS]
    if method == 'stem':
        return ' '.join([stemmer.stem(w) for w in words])
    else:
        return ' '.join([lemmatizer.lemmatize(w) for w in words])

# @app.route("/predict_random", methods=["GET"])
# def predict_random():
#     tweet = df['text'].sample(1).values[0]
#     cleaned = preprocess(tweet)
#     vect = vectorizer.transform([cleaned])
#     score = model.predict_proba(vect)[0][1]
#     sentiment = "positif" if score >= 0.5 else "négatif"

# 📲 Endpoint JSON
@app.route("/predict_json", methods=["POST"])
def predict_json():
    data = request.get_json()
    tweet = data.get("tweet", "")
    # cleaned = preprocess(tweet)
# Use advanced cleaning (lemmatization by default)
    cleaned = clean_text(tweet, method='lemma')
    vect = vectorizer.transform([cleaned])
    score = model.predict_proba(vect)[0][1]
    sentiment = "positif" if score >= 0.5 else "négatif"

    # 📦 Tracking MLflow (inférences)
    # mlflow.set_tracking_uri("http://localhost:5400")
    # mlflow.set_experiment("tweet_sentiment")
    # with mlflow.start_run(run_name="inference", nested=True):
    #     mlflow.log_param("model_used", "TF-IDF + LogisticRegression")
    #     # mlflow.log_input(pd.DataFrame([{"tweet": tweet}]))
    #     mlflow.log_metric("score", score)

    return jsonify({
        "tweet": tweet,
        "prediction": sentiment,
        "score": round(float(score), 4)
    })

@app.route("/log_feedback", methods=["POST"])
def log_feedback():
    data = request.get_json()
    logger.info("User feedback", extra={"custom_dimensions": data})
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run()
    # app.run(debug=True, port=5500)
# Ajouter tests unitaires !!!