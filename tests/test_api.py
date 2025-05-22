import pytest
import importlib.util
# from app_with_mlflow import app
import sys
import os

# 📍 Localisation absolue du fichier app_with_mlflow.py
API_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app_with_mlflow.py'))

# 📦 Import dynamique du module
spec = importlib.util.spec_from_file_location("app_with_mlflow", API_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# ✅ Accès à l'app Flask
app = module.app
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_json_valid_input(client):
    """Test de prédiction avec un texte valide."""
    response = client.post('/predict_json', json={"tweet": "J'adore ce service aérien !"})
    assert response.status_code == 200
    data = response.get_json()
    assert "tweet" in data
    assert "prediction" in data
    assert data["prediction"] in ["positif", "négatif"]
    assert isinstance(data["score"], float)

def test_predict_json_empty_input(client):
    """Test avec un texte vide."""
    response = client.post('/predict_json', json={"tweet": ""})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"] in ["positif", "négatif"]

def test_predict_json_missing_key(client):
    """Test avec un JSON sans la clé 'tweet'."""
    response = client.post('/predict_json', json={})
    assert response.status_code == 200
    data = response.get_json()
    assert "tweet" in data
    assert "prediction" in data

def test_predict_json_wrong_method(client):
    """Test d'appel en GET au lieu de POST."""
    response = client.get('/predict_json')
    assert response.status_code == 405  # Method Not Allowed

def test_predict_json_non_json_payload(client):
    """Test d'envoi de données non-JSON."""
    response = client.post('/predict_json', data="Ce n'est pas du JSON", content_type="text/plain")
    assert response.status_code in (400, 415, 500)  # Cela dépend de Flask si c'est géré ou non

def test_predict_json_unicode_input(client):
    """Test avec caractères unicode."""
    response = client.post('/predict_json', json={"tweet": "C’était génial 😍✨"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] in ["positif", "négatif"]

def test_predict_json_special_characters(client):
    """Test avec ponctuations et symboles spéciaux."""
    response = client.post('/predict_json', json={"tweet": "@AirParadis #super ✈️!! https://ex.com"})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
