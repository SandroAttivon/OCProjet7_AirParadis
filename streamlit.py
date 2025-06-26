import streamlit as st 
import requests
import mlflow
import json
from datetime import datetime
# import os
# from app import app  # Import de l'application Flask

# port = int(os.environ.get("PORT", 8501))  # Azure injecte PORT (ex: 8000 ou 80)
# app.run(host="0.0.0.0", port=port)


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

# Helper: store prediction result in session_state
def store_prediction(result):
    st.session_state['prediction_result'] = result
    st.session_state['show_feedback'] = True
    st.session_state['feedback'] = 'Oui'

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
                store_prediction(result)
            else:
                st.error(f"❌ Erreur {response.status_code} : {response.text}")
        except Exception as e:
            st.error(f"⚠️ Erreur de connexion à l'API : {e}")
    else:
        st.warning("Veuillez entrer un texte.")

# Show feedback UI if prediction was made
if st.session_state.get('show_feedback', False):
    result = st.session_state['prediction_result']
    with st.form(key="feedback_form"):
        feedback = st.radio(
            "🧠 Ce résultat est-il correct ?",
            ("Oui", "Non"),
            horizontal=True,
            index=("Oui", "Non").index(st.session_state.get('feedback', 'Oui'))
        )
        submit_feedback = st.form_submit_button("📤 Envoyer le feedback")
        if submit_feedback:
            now = datetime.utcnow().isoformat()
            log_data = {
                "timestamp": now,
                "tweet_text": result["tweet"],
                "prediction": result["prediction"],
                "score": round(result["score"], 4),
                "user_feedback": feedback
            }
            # Send feedback to Flask API
            try:
                feedback_api_url = API_URL.replace("/predict_json", "/log_feedback")
                resp = requests.post(feedback_api_url, json=log_data)
                if resp.status_code == 200:
                    st.success("🎉 Feedback enregistré dans Application Insights !")
                else:
                    st.error(f"Erreur lors de l'envoi du feedback : {resp.text}")
            except Exception as e:
                st.error(f"Erreur de connexion à l'API feedback : {e}")
            st.session_state['show_feedback'] = False
            st.session_state['feedback'] = feedback
