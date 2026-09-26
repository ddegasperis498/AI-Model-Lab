from __future__ import annotations
import json

from .schemas_v4 import V4LMConfig
from .trainer_v4 import UltimateLMTrainer
from .profiler_v4 import analytical_decoder_estimate


def main():
    corpus = ("AI Model Lab impara token, attention, gradienti e ottimizzazione. " * 20)
    cfg = V4LMConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, context_length=32,
        mlp_ratio=2.0, dropout=0.0, tokenizer='bpe', bpe_vocab_size=280,
        optimizer='adamw', lr=1e-3, min_lr=1e-4, warmup_steps=1,
        lr_decay_steps=20, weight_decay=0.01, grad_clip=1.0,
        grad_accum_steps=1, device='cpu', amp_dtype='none', seed=7,
        lora_rank=4, lora_alpha=8.0, lora_freeze_base=False,
    )
    t = UltimateLMTrainer(cfg, corpus, validation_fraction=0.1)
    before = t.state()['params']
    result = t.train(steps=2, batch_size=2, eval_interval=1)
    gen = t.generate('AI ', max_new_tokens=8, temperature=1.0, top_k=20, top_p=0.9, repetition_penalty=1.0, seed=1)
    preview = t.tokenizer_preview('AI Model Lab', 20)
    sim = t.embedding_similarity('AI Model', 8)
    est = analytical_decoder_estimate(32000, 512, 8, 2, 8, 1024, 4.0, 1, 2, 'adamw')
    assert t.step == 2
    assert len(gen['token_ids']) > 0
    assert preview
    assert sim['matrix']
    assert est['parameter_count_estimate'] > 0
    print(json.dumps({
        'ok': True,
        'params': before,
        'loss': result['last']['train_loss'],
        'val_loss': result['last'].get('val_loss'),
        'tokenizer_vocab': t.tokenizer.vocab_size,
        'generated': gen['text'][:80],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
