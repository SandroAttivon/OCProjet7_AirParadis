from flask import Flask, request, jsonify
import pandas as pd
import joblib
import re
import mlflow
# import os

app = Flask(__name__)


# port = int(os.environ.get("PORT", 5500))  # Azure injecte PORT (ex: 8000 ou 80)
# app.run(host="0.0.0.0", port=port)

# 📂 Chemins vers les modèles et données
input_path = 'C:/Users/sandr/OneDrive/Documents/JOB/OPENCLASSROOMS/AI_ENGINEER/Projet_7_Réalisez_une_analyse_de_sentiments_grâce_au_Deep_Learning/Workspace/'
MODEL_PATH = input_path + "models/model_logreg.pkl"
VECTORIZER_PATH = input_path + "models/tfidf_vectorizer.pkl"
# CSV_PATH = input_path + "data/raw/training.1600000.processed.noemoticon.csv"

# 🔁 Chargement du modèle et du vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# 📄 Chargement du fichier CSV une seule fois
# df = pd.read_csv(CSV_PATH, encoding="latin-1", header=None)
# df.columns = ["sentiment", "id", "date", "flag", "user", "text"]

# 🧹 Fonction de nettoyage
def preprocess(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#", '', text)
    text = re.sub(r"[^\w\s]", '', text.lower())
    return text

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
    cleaned = preprocess(tweet)
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

if __name__ == "__main__":
    app.run(debug=True, port=5500)
# Ajouter tests unitaires !!!