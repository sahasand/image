from fastapi.testclient import TestClient
import sys
import types
from pathlib import Path

# Provide lightweight stubs for optional heavy dependencies so that
# the API module can be imported without installing the actual packages.
diffusers_stub = types.ModuleType("diffusers")

class DummyPipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        return types.SimpleNamespace(images=[])


diffusers_stub.StableDiffusionPipeline = DummyPipeline
sys.modules.setdefault("diffusers", diffusers_stub)

torch_stub = types.ModuleType("torch")
torch_stub.float16 = "float16"
torch_stub.float32 = "float32"

class DummyGenerator:
    def manual_seed(self, *args, **kwargs):  # pragma: no cover - simple stub
        return self


torch_stub.Generator = lambda *args, **kwargs: DummyGenerator()
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_stub.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
sys.modules.setdefault("torch", torch_stub)

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


def test_generate_with_negative_prompt(monkeypatch):
    """Ensure negative prompts are passed through to the generator."""
    captured = {}

    class DummyGen:
        def generate_image(self, *args, **kwargs):
            captured['negative_prompt'] = kwargs.get('negative_prompt')
            return {"images": ["ZW1wdHk="], "image_paths": ["/tmp/a.png"]}

    from . import api as api_module
    monkeypatch.setattr(api_module, 'generator', DummyGen())
    res = client.post(
        '/generate', json={'prompt': 'x', 'negative_prompt': 'y'}
    )
    assert res.status_code == 200
    assert res.json()['success']
    assert captured['negative_prompt'] == 'y'


def test_models_list_and_switch(monkeypatch):
    """Test model listing, switching and current model reporting."""
    from . import api as api_module
    from backend.config import config as config_module

    monkeypatch.setattr(config_module, 'MODELS', {'a': object(), 'b': object()})

    class DummyGen:
        def __init__(self):
            self.current_model = None

        def load_model(self, model_name):
            if model_name not in config_module.MODELS:
                raise ValueError('bad model')
            self.current_model = model_name

    dummy = DummyGen()
    monkeypatch.setattr(api_module, 'generator', dummy)

    res = client.get('/models')
    assert res.status_code == 200
    assert res.json() == ['a', 'b']

    res = client.get('/models/current')
    assert res.status_code == 200
    assert res.json()['current_model'] is None

    res = client.post('/models/switch/b')
    assert res.status_code == 200
    assert res.json()['model'] == 'b'
    assert dummy.current_model == 'b'

    res = client.get('/models/current')
    assert res.json()['current_model'] == 'b'


def test_invalid_model_error(monkeypatch):
    """Ensure switching to an invalid model returns an error."""
    from . import api as api_module

    class DummyGen:
        def load_model(self, name):
            raise ValueError('Unknown model')

    monkeypatch.setattr(api_module, 'generator', DummyGen())
    local_client = TestClient(app, raise_server_exceptions=False)
    res = local_client.post('/models/switch/doesnotexist')
    assert res.status_code == 500


