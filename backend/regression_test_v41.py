from __future__ import annotations

import math
from fastapi.testclient import TestClient

from backend.safe_json import sanitize_for_json
from backend.server import app


def assert_json_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), f'Non-finite float leaked into JSON: {value!r}'
    elif isinstance(value, dict):
        for v in value.values():
            assert_json_finite(v)
    elif isinstance(value, list):
        for v in value:
            assert_json_finite(v)


assert sanitize_for_json(float('nan')) is None
assert sanitize_for_json(float('inf')) is None
assert sanitize_for_json(float('-inf')) is None
assert sanitize_for_json({'x': [1.0, float('nan')]}) == {'x': [1.0, None]}

client = TestClient(app)

health = client.get('/api/advanced/health')
assert health.status_code == 200, health.text

# Reproduce the exact MLP configuration that previously produced a JSON 500.
config = {
    'input_size': 1,
    'hidden_sizes': [32, 32],
    'output_size': 1,
    'task': 'regression',
    'activation': 'relu',
    'dropout': 0.0,
    'optimizer': 'adamw',
    'lr': 0.001,
    'weight_decay': 0.0001,
    'grad_clip': 1.0,
    'seed': 42,
    'device': 'cpu',
    'amp': False,
}
create = client.post('/api/advanced/mlp/create', json=config)
assert create.status_code == 200, create.text
assert create.json()['params']['total'] == 1153

samples = []
for i in range(160):
    x = -3.0 + 6.0 * i / 159.0
    samples.append({'x': [x], 'y': [2.5 * x + 1.0]})

one = client.post('/api/advanced/mlp/train', json={'data': samples, 'batch_size': 32, 'steps': 1})
assert one.status_code == 200, one.text
payload = one.json()
assert payload['step'] == 1
assert payload['device'] == 'cpu'
assert_json_finite(payload)
assert payload['last']['loss'] is not None

fifty = client.post('/api/advanced/mlp/train', json={'data': samples, 'batch_size': 32, 'steps': 50})
assert fifty.status_code == 200, fifty.text
payload50 = fifty.json()
assert payload50['step'] == 51
assert_json_finite(payload50)
assert len(payload50.get('loss_curve', [])) == 50


# CUDA/AMP regression: reproduce the exact sequence that previously left
# GradScaler in UNSCALED state (1 step request followed by a 50-step request).
if health.json().get('cuda_available'):
    cuda_config = dict(config)
    cuda_config['device'] = 'cuda'
    cuda_config['amp'] = True
    cuda_create = client.post('/api/advanced/mlp/create', json=cuda_config)
    assert cuda_create.status_code == 200, cuda_create.text

    cuda_one = client.post('/api/advanced/mlp/train', json={'data': samples, 'batch_size': 32, 'steps': 1})
    assert cuda_one.status_code == 200, cuda_one.text
    assert cuda_one.json()['step'] == 1
    assert cuda_one.json()['device'] == 'cuda'
    assert_json_finite(cuda_one.json())

    cuda_fifty = client.post('/api/advanced/mlp/train', json={'data': samples, 'batch_size': 32, 'steps': 50})
    assert cuda_fifty.status_code == 200, cuda_fifty.text
    assert cuda_fifty.json()['step'] == 51
    assert_json_finite(cuda_fifty.json())
    assert len(cuda_fifty.json().get('loss_curve', [])) == 50
else:
    print('CUDA/AMP regression: SKIPPED (CUDA non disponibile)')

# Error responses must use the structured error contract.
bad = client.post('/api/advanced/mlp/train', json={'data': [], 'batch_size': 32, 'steps': 1})
assert bad.status_code in (400, 422), bad.text
if bad.status_code == 400:
    err = bad.json()
    assert err['ok'] is False
    assert 'error' in err
    assert err['error']['correlation_id']

print('AI Model Lab V4.1 regression tests: OK')
print('PyTorch:', health.json()['pytorch'])
print('MLP params:', create.json()['params']['total'])
print('Step 1 loss:', one.json()['last']['loss'])
print('Step 51 loss:', payload50['last']['loss'])
