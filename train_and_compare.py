
import pandas as pd
import joblib
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

# 📁 Chemins
DATA_PATH = "data/tweets.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 📄 Chargement des données
df = pd.read_csv(DATA_PATH)

# ✅ Nettoyage basique
df['text_clean'] = df['text'].str.lower().apply(lambda x: re.sub(r"[^\w\s]", "", x))

# 🎯 Label fictif si non présent
if 'sentiment' not in df.columns:
    print("⚠️ Aucun label 'sentiment' trouvé, une colonne factice est utilisée (tout à 0)")
    df['sentiment'] = 0

# ✂️ Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text_clean'])
y = df['sentiment']

# 🔀 Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 📊 Modèles à comparer
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

results = []

# 🔁 Entraînement + Évaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    results.append((name, acc, f1, recall))
    print(f"✔️ {name}: Accuracy={acc:.4f} | F1={f1:.4f} | Recall={recall:.4f}")

# 🥇 Meilleur modèle (par F1-score)
best_model_name, *_ = max(results, key=lambda x: x[2])
best_model = models[best_model_name]

# 💾 Sauvegarde du meilleur
joblib.dump(best_model, os.path.join(MODEL_DIR, "model_logreg.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print(f"✅ Modèle sélectionné : {best_model_name} (sauvegardé dans 'model/')")
