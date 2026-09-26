from __future__ import annotations

import math
from fastapi.testclient import TestClient

from backend.safe_json import sanitize_for_json
from backend.server import app, CHECKPOINTS


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

# Checkpoint regression: a resume must restore the step counter and the
# last visible metrics together with model/optimizer/scaler state.
saved_loss = payload50['last']['loss']
save = client.post('/api/advanced/checkpoint/save', json={'name': 'regression_v413_resume'})
assert save.status_code == 200, save.text
checkpoint_name = save.json()['filename']

advance = client.post('/api/advanced/mlp/train', json={'data': samples, 'batch_size': 32, 'steps': 1})
assert advance.status_code == 200, advance.text
assert advance.json()['step'] == 52

loaded = client.post('/api/advanced/checkpoint/load', json={'filename': checkpoint_name})
assert loaded.status_code == 200, loaded.text
loaded_payload = loaded.json()
assert loaded_payload['step'] == 51
assert loaded_payload['last']['loss'] == saved_loss
assert_json_finite(loaded_payload)
checkpoint_path = CHECKPOINTS / checkpoint_name
if checkpoint_path.exists():
    checkpoint_path.unlink()

# Transformer Validation Lab regression.
llm_config = {
    'd_model': 32,
    'n_heads': 4,
    'n_layers': 1,
    'context_length': 16,
    'mlp_ratio': 2.0,
    'dropout': 0.0,
    'optimizer': 'adamw',
    'lr': 0.001,
    'weight_decay': 0.0,
    'grad_clip': 1.0,
    'seed': 42,
    'device': 'cpu',
    'amp': False,
    'tie_embeddings': True,
}
llm_create = client.post('/api/advanced/llm/create', json=llm_config)
assert llm_create.status_code == 200, llm_create.text

corpus = (
    "Il modello impara dai dati e deve prevedere il token successivo. "
    "La validation misura la generalizzazione su dati non usati per aggiornare i pesi. "
) * 6
llm_train = client.post('/api/advanced/llm/train', json={
    'text': corpus,
    'batch_size': 4,
    'steps': 3,
    'validation_fraction': 0.20,
})
assert llm_train.status_code == 200, llm_train.text
llm_payload = llm_train.json()
assert llm_payload['step'] == 3
assert len(llm_payload['train_loss_curve']) == 3
assert len(llm_payload['val_loss_curve']) == 3
assert llm_payload['last']['train_loss'] is not None
assert llm_payload['last']['val_loss'] is not None
assert llm_payload['last']['train_perplexity'] is not None
assert llm_payload['last']['val_perplexity'] is not None
assert llm_payload['last']['generalization_gap'] is not None
assert llm_payload['last']['generalization_status'] in ('warming_up', 'learning', 'overfitting', 'underfitting', 'stable')
assert abs(llm_payload['validation']['fraction'] - 0.20) < 1e-9
assert_json_finite(llm_payload)

gen_body = {
    'prompt': 'Il modello ',
    'max_new_tokens': 24,
    'temperature': 0.8,
    'top_k': 20,
    'seed': 1234,
}
gen_a = client.post('/api/advanced/llm/generate', json=gen_body)
gen_b = client.post('/api/advanced/llm/generate', json=gen_body)
assert gen_a.status_code == 200, gen_a.text
assert gen_b.status_code == 200, gen_b.text
assert gen_a.json()['token_ids'] == gen_b.json()['token_ids']
assert gen_a.json()['text'] == gen_b.json()['text']
assert gen_a.json()['seed'] == 1234

# LLM checkpoint must preserve validation history and visible metrics.
llm_saved_loss = llm_payload['last']['val_loss']
llm_save = client.post('/api/advanced/checkpoint/save', json={'name': 'regression_v420_llm'})
assert llm_save.status_code == 200, llm_save.text
llm_checkpoint_name = llm_save.json()['filename']
llm_advance = client.post('/api/advanced/llm/train', json={
    'text': corpus, 'batch_size': 4, 'steps': 1, 'validation_fraction': 0.20,
})
assert llm_advance.status_code == 200, llm_advance.text
assert llm_advance.json()['step'] == 4
llm_loaded = client.post('/api/advanced/checkpoint/load', json={'filename': llm_checkpoint_name})
assert llm_loaded.status_code == 200, llm_loaded.text
assert llm_loaded.json()['step'] == 3
assert llm_loaded.json()['last']['val_loss'] == llm_saved_loss
assert llm_loaded.json()['validation']['history_points'] == 3
llm_checkpoint_path = CHECKPOINTS / llm_checkpoint_name
if llm_checkpoint_path.exists():
    llm_checkpoint_path.unlink()

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

print('AI Model Lab V4.2 regression tests: OK')
print('PyTorch:', health.json()['pytorch'])
print('MLP params:', create.json()['params']['total'])
print('Step 1 loss:', one.json()['last']['loss'])
print('Step 51 loss:', payload50['last']['loss'])

print('Transformer validation loss:', llm_payload['last']['val_loss'])
print('Deterministic generation seed:', gen_a.json()['seed'])
