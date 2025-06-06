import streamlit as st 
import requests
import mlflow
import json
from datetime import datetime

# 📌 Configuration de la page
st.set_page_config(page_title="Analyse de sentiment", page_icon="✈️")
st.title("✈️ Prédiction de sentiment - Air Paradis")
st.markdown("Entrez un tweet ou un message pour analyser son **sentiment**.")

# 🌐 Configuration API et MLflow
API_URL = "http://127.0.0.1:5500/predict_json"
# mlflow.set_tracking_uri("http://localhost:5400")
# mlflow.set_experiment("tweet_sentiment")

# 📝 Zone de saisie utilisateur
user_input = st.text_area("Saisissez un texte ici :", height=100)

# 🔍 Analyse du sentiment
if st.button("🔍 Analyser le sentiment"):
    if user_input.strip():
        try:
            response = requests.post(API_URL, json={"tweet": user_input})
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Prédiction réussie !")
                st.markdown(f"**✍️ Texte :**\n> {result['tweet']}")
                st.markdown(f"**🔍 Sentiment :** `{result['prediction']}`")
                st.markdown(f"**📊 Score :** {round(result['score'], 4)}")

                # 🔘 Feedback utilisateur
                feedback = st.radio("🧠 Ce résultat est-il correct ?", ("Oui", "Non"), horizontal=True)

                if st.button("📤 Envoyer le feedback"):
                    now = datetime.utcnow().isoformat()
                    log_data = {
                        "timestamp": now,
                        "tweet_text": result["tweet"],
                        "prediction": result["prediction"],
                        "score": round(result["score"], 4),
                        "user_feedback": feedback
                    }

                    # 💾 Sauvegarde temporaire locale
                    with open("feedback_details.json", "w", encoding="utf-8") as f:
                        json.dump(log_data, f, indent=2, ensure_ascii=False)

                    # 🚀 Logging dans MLflow
                    with mlflow.start_run(run_name="user_feedback", nested=True):
                        mlflow.log_param("model_used", "TF-IDF + LogisticRegression")
                        mlflow.log_param("user_validation", feedback)
                        mlflow.log_metric("score", log_data["score"])
                        mlflow.set_tag("prediction", result["prediction"])
                        mlflow.set_tag("timestamp", now)
                        mlflow.log_artifact("feedback_details.json")

                    st.success("🎉 Feedback enregistré dans MLflow.")
            else:
                st.error(f"❌ Erreur {response.status_code} : {response.text}")
        except Exception as e:
            st.error(f"⚠️ Erreur de connexion à l'API : {e}")
    else:
        st.warning("Veuillez entrer un texte.")
