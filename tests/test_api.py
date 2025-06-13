import pytest
from source.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_json_positive(client):
    response = client.post("/predict_json", json={"tweet": "I love this airline, great service!"})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"] in ["positif", "négatif"]
    assert isinstance(data["score"], float)

def test_predict_json_negative(client):
    response = client.post("/predict_json", json={"tweet": "Worst flight ever, very disappointed."})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"] in ["positif", "négatif"]
    assert isinstance(data["score"], float)

def test_predict_json_empty(client):
    response = client.post("/predict_json", json={"tweet": ""})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"] in ["positif", "négatif"]
    assert isinstance(data["score"], float)