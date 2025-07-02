from fastapi.testclient import TestClient
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.src.api import app

client = TestClient(app)

def test_health():
    res = client.get('/health')
    assert res.status_code == 200
    assert res.json()['status'] == 'ok'

def test_generate_minimal(monkeypatch):
    class DummyGen:
        def generate_image(self, *args, **kwargs):
            return {"images": ["dGVzdA=="], "image_paths": ["/tmp/test.png"]}
    from . import api as api_module
    monkeypatch.setattr(api_module, 'generator', DummyGen())
    res = client.post('/generate', json={'prompt': 'test'})
    assert res.status_code == 200
    assert res.json()['success']


