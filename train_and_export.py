import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 📁 Chemins
input_path = 'C:/Users/sandr/OneDrive/Documents/JOB/OPENCLASSROOMS/AI_ENGINEER/Projet_7_Réalisez_une_analyse_de_sentiments_grâce_au_Deep_Learning/Workspace/'
DATA_PATH = input_path + "data/raw/training.1600000.processed.noemoticon.csv"
MODEL_DIR = input_path + "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# 📄 Chargement des données
df = pd.read_csv(DATA_PATH, encoding="latin-1", header=None)
df.columns = ["sentiment", "id", "date", "flag", "user", "text"]

# ✅ Nettoyage basique
df['text_clean'] = df['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# 🎯 Label fictif si non présent
if 'sentiment' not in df.columns:
    print("⚠️ Aucun label 'sentiment' trouvé, une colonne factice est utilisée (tout à 0)")
    df['sentiment'] = 0

# ✂️ Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text_clean'])
y = df['sentiment']

# 🎓 Entraînement du modèle
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 💾 Sauvegarde
joblib.dump(model, os.path.join(MODEL_DIR, "model_logreg.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("✅ Modèle et vectorizer sauvegardés dans 'models/'")
