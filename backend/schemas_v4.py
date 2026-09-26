from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

DeviceChoice = Literal['auto', 'cpu', 'cuda']
OptimizerName = Literal['sgd', 'adam', 'adamw']
TokenizerKind = Literal['byte', 'bpe']
AmpDType = Literal['auto', 'none', 'fp16', 'bf16']


class V4LMConfig(BaseModel):
    d_model: int = Field(192, ge=32, le=4096)
    n_heads: int = Field(6, ge=1, le=64)
    n_kv_heads: int = Field(2, ge=1, le=64)
    n_layers: int = Field(6, ge=1, le=96)
    context_length: int = Field(256, ge=16, le=8192)
    mlp_ratio: float = Field(4.0, ge=1.0, le=16.0)
    dropout: float = Field(0.05, ge=0.0, lt=1.0)
    rope_theta: float = Field(10000.0, ge=100.0, le=10_000_000.0)
    bias: bool = False
    tie_embeddings: bool = True
    use_sdpa: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False

    tokenizer: TokenizerKind = 'byte'
    bpe_vocab_size: int = Field(384, ge=256, le=4096)

    lora_rank: int = Field(0, ge=0, le=256)
    lora_alpha: float = Field(16.0, ge=0.1, le=1024.0)
    lora_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    lora_freeze_base: bool = False

    optimizer: OptimizerName = 'adamw'
    lr: float = Field(3e-4, gt=0.0, le=1.0)
    min_lr: float = Field(3e-5, ge=0.0, le=1.0)
    warmup_steps: int = Field(20, ge=0, le=1_000_000)
    lr_decay_steps: int = Field(2000, ge=1, le=100_000_000)
    weight_decay: float = Field(0.1, ge=0.0, le=10.0)
    grad_clip: float = Field(1.0, gt=0.0, le=1000.0)
    grad_accum_steps: int = Field(1, ge=1, le=1024)
    seed: int = 42
    device: DeviceChoice = 'auto'
    amp_dtype: AmpDType = 'auto'

    @model_validator(mode='after')
    def validate_heads(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model deve essere divisibile per n_heads')
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError('n_heads deve essere divisibile per n_kv_heads')
        if self.min_lr > self.lr:
            raise ValueError('min_lr non può essere maggiore di lr')
        return self


class V4CreateRequest(BaseModel):
    config: V4LMConfig
    corpus: str = Field(min_length=64)
    validation_fraction: float = Field(0.1, ge=0.0, le=0.45)


class V4TrainRequest(BaseModel):
    steps: int = Field(1, ge=1, le=5000)
    batch_size: int = Field(8, ge=1, le=1024)
    eval_interval: int = Field(20, ge=1, le=10000)


class V4JobRequest(BaseModel):
    steps: int = Field(200, ge=1, le=5_000_000)
    batch_size: int = Field(8, ge=1, le=1024)
    eval_interval: int = Field(20, ge=1, le=10000)


class V4GenerateRequest(BaseModel):
    prompt: str = ''
    max_new_tokens: int = Field(100, ge=1, le=4096)
    temperature: float = Field(0.8, gt=0.0, le=5.0)
    top_k: int = Field(50, ge=0, le=4096)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.05, ge=0.5, le=3.0)
    seed: Optional[int] = None


class V4TokenizerPreviewRequest(BaseModel):
    text: str = ''
    limit: int = Field(80, ge=1, le=512)


class V4ExperimentSaveRequest(BaseModel):
    name: str = Field('experiment', min_length=1, max_length=120)
    notes: str = Field('', max_length=4000)


class V4CheckpointRequest(BaseModel):
    name: str = Field('ultimate', min_length=1, max_length=120)


class V4CheckpointLoadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=200)


class V4ScaleRequest(BaseModel):
    vocab_size: int = Field(32000, ge=256, le=1_000_000)
    d_model: int = Field(1024, ge=32, le=32768)
    n_heads: int = Field(16, ge=1, le=256)
    n_kv_heads: int = Field(4, ge=1, le=256)
    n_layers: int = Field(24, ge=1, le=256)
    context_length: int = Field(4096, ge=16, le=1_000_000)
    mlp_ratio: float = Field(4.0, ge=1.0, le=16.0)
    batch_size: int = Field(1, ge=1, le=100000)
    precision_bytes: int = Field(2, ge=1, le=8)
    optimizer: Literal['none', 'sgd', 'adamw'] = 'adamw'
