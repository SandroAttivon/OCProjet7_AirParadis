#!/bin/bash

echo "🔧 Initialisation du projet AirParadis Sentiment..."

# Création de l'arborescence
mkdir -p data/raw data/processed notebooks src models api app tests .github/workflows insights

# Création du README et requirements.txt s'ils n'existent pas
touch README.md
touch requirements.txt

# Initialisation du dépôt Git
git init

# Création de l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Mise à jour de pip
pip install --upgrade pip

# Installation des dépendances
if [ -s requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "⚠️  Le fichier requirements.txt est vide. Tu dois le remplir avant d’installer les packages."
fi

# Création du fichier .gitignore
cat <<EOL > .gitignore
__pycache__/
*.pyc
mlruns/
models/
.env
*.log
.ipynb_checkpoints
.venv/
EOL

echo "✅ Projet initialisé avec succès."
echo "➡️  Active ton environnement avec : source .venv/bin/activate"
