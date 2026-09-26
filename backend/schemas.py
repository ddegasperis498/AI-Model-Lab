from __future__ import annotations
from typing import Literal, List, Optional
from pydantic import BaseModel, Field

DeviceChoice = Literal['auto', 'cpu', 'cuda']
OptimizerName = Literal['sgd', 'adam', 'adamw']
ActivationName = Literal['relu', 'gelu', 'tanh', 'sigmoid']
TaskName = Literal['regression', 'classification']

class Sample(BaseModel):
    x: List[float]
    y: List[float]

class MLPConfig(BaseModel):
    input_size: int = Field(1, ge=1, le=128)
    hidden_sizes: List[int] = Field(default_factory=lambda: [32, 32])
    output_size: int = Field(1, ge=1, le=128)
    task: TaskName = 'regression'
    activation: ActivationName = 'relu'
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    optimizer: OptimizerName = 'adamw'
    lr: float = Field(1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(1e-4, ge=0.0, le=1.0)
    grad_clip: float = Field(1.0, gt=0.0, le=1000.0)
    seed: int = 42
    device: DeviceChoice = 'auto'
    amp: bool = True

class MLPTrainRequest(BaseModel):
    data: List[Sample]
    batch_size: int = Field(16, ge=1, le=4096)
    steps: int = Field(1, ge=1, le=1000)

class MLPPredictRequest(BaseModel):
    x: List[List[float]]

class LLMConfig(BaseModel):
    d_model: int = Field(128, ge=32, le=2048)
    n_heads: int = Field(4, ge=1, le=32)
    n_layers: int = Field(4, ge=1, le=48)
    context_length: int = Field(128, ge=16, le=4096)
    mlp_ratio: float = Field(4.0, ge=1.0, le=16.0)
    dropout: float = Field(0.1, ge=0.0, lt=1.0)
    optimizer: OptimizerName = 'adamw'
    lr: float = Field(3e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(0.01, ge=0.0, le=1.0)
    grad_clip: float = Field(1.0, gt=0.0, le=1000.0)
    seed: int = 42
    device: DeviceChoice = 'auto'
    amp: bool = True
    tie_embeddings: bool = True

class LLMTrainRequest(BaseModel):
    text: str = Field(min_length=32)
    batch_size: int = Field(8, ge=1, le=512)
    steps: int = Field(1, ge=1, le=500)

class LLMGenerateRequest(BaseModel):
    prompt: str = ''
    max_new_tokens: int = Field(80, ge=1, le=2048)
    temperature: float = Field(0.8, gt=0.0, le=5.0)
    top_k: int = Field(40, ge=0, le=256)

class CheckpointRequest(BaseModel):
    name: str = Field('checkpoint', min_length=1, max_length=80)

class CheckpointLoadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=160)
