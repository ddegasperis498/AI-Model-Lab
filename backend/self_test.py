from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)

health = client.get('/api/advanced/health')
assert health.status_code == 200

mlp_cfg = {
    'input_size': 1, 'hidden_sizes': [8, 8], 'output_size': 1,
    'task': 'regression', 'activation': 'tanh', 'dropout': 0.0,
    'optimizer': 'adamw', 'lr': 0.01, 'weight_decay': 0.0,
    'grad_clip': 1.0, 'seed': 42, 'device': 'cpu', 'amp': False,
}
assert client.post('/api/advanced/mlp/create', json=mlp_cfg).status_code == 200
samples = [{'x': [x/10], 'y': [2.5*(x/10)+1]} for x in range(-20, 21)]
result = client.post('/api/advanced/mlp/train', json={'data': samples, 'batch_size': 16, 'steps': 4})
assert result.status_code == 200

llm_cfg = {
    'd_model': 32, 'n_heads': 4, 'n_layers': 1, 'context_length': 16,
    'mlp_ratio': 2.0, 'dropout': 0.0, 'optimizer': 'adamw',
    'lr': 0.001, 'weight_decay': 0.01, 'grad_clip': 1.0,
    'seed': 42, 'device': 'cpu', 'amp': False, 'tie_embeddings': True,
}
assert client.post('/api/advanced/llm/create', json=llm_cfg).status_code == 200
text = ('ciao mondo questo e un modello transformer reale. ') * 20
result = client.post('/api/advanced/llm/train', json={'text': text, 'batch_size': 2, 'steps': 1})
assert result.status_code == 200
result = client.post('/api/advanced/llm/generate', json={'prompt': 'ciao ', 'max_new_tokens': 4, 'temperature': 1.0, 'top_k': 20})
assert result.status_code == 200

print('AI Model Lab V3 backend self-test: OK')
print('PyTorch:', health.json()['pytorch'])
print('Device:', health.json()['device_name'])
