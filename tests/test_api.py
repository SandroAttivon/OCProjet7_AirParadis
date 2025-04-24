import unittest
import json
from app import app  # Assure-toi que app.py est bien au niveau racine ou adapte l'import

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.url = '/predict'

    def test_prediction_positive(self):
        response = self.app.post(self.url,
                                 data=json.dumps({"text": "I love this airline!"}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("sentiment", data)
        self.assertIn(data["sentiment"], ["positif", "négatif"])

    def test_prediction_missing_text(self):
        response = self.app.post(self.url,
                                 data=json.dumps({}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

if __name__ == '__main__':
    unittest.main()
