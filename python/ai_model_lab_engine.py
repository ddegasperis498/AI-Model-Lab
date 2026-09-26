"""
AI Model Lab - pure Python educational MLP engine.
No third-party dependencies are required.

Architecture:
    input -> Dense hidden layers -> output

Supported:
    - tanh / ReLU / sigmoid / linear
    - regression (MSE)
    - binary classification (binary cross entropy)
    - manual backpropagation
    - SGD
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Callable

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid(x: float) -> float:
    x = clamp(x, -40.0, 40.0)
    return 1.0 / (1.0 + math.exp(-x))

ACT = {
    "linear": (lambda x: x, lambda z, a: 1.0),
    "tanh": (math.tanh, lambda z, a: 1.0 - a*a),
    "relu": (lambda x: max(0.0, x), lambda z, a: 1.0 if z > 0 else 0.0),
    "sigmoid": (sigmoid, lambda z, a: a * (1.0 - a)),
}

@dataclass
class Dense:
    in_size: int
    out_size: int
    activation: str = "tanh"
    W: List[List[float]] = field(init=False)
    b: List[float] = field(init=False)
    cache: dict = field(default_factory=dict, init=False)
    grads: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        scale = math.sqrt(2.0 / max(1, self.in_size))
        self.W = [
            [random.uniform(-scale, scale) for _ in range(self.in_size)]
            for _ in range(self.out_size)
        ]
        self.b = [random.uniform(-0.08, 0.08) for _ in range(self.out_size)]

    def forward(self, x: List[float]) -> List[float]:
        f, _ = ACT[self.activation]
        z = [
            sum(w * xi for w, xi in zip(row, x)) + self.b[j]
            for j, row in enumerate(self.W)
        ]
        a = [f(v) for v in z]
        self.cache = {"x": x[:], "z": z, "a": a}
        return a

    def backward(self, grad_out: List[float]) -> List[float]:
        x, z, a = self.cache["x"], self.cache["z"], self.cache["a"]
        _, derivative = ACT[self.activation]

        dz = [g * derivative(zj, aj) for g, zj, aj in zip(grad_out, z, a)]
        dW = [[dz[j] * x[i] for i in range(self.in_size)] for j in range(self.out_size)]
        db = dz[:]

        grad_input = [0.0 for _ in range(self.in_size)]
        for j in range(self.out_size):
            for i in range(self.in_size):
                grad_input[i] += self.W[j][i] * dz[j]

        self.grads = {"dW": dW, "db": db}
        return grad_input

    def update_sgd(self, lr: float):
        for j in range(self.out_size):
            for i in range(self.in_size):
                self.W[j][i] -= lr * clamp(self.grads["dW"][j][i], -20, 20)
            self.b[j] -= lr * clamp(self.grads["db"][j], -20, 20)

class MLP:
    def __init__(self, sizes: List[int], hidden_activation="tanh", task="regression"):
        self.sizes = sizes
        self.task = task
        self.layers: List[Dense] = []

        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            activation = ("sigmoid" if task == "classification" else "linear") if last else hidden_activation
            self.layers.append(Dense(sizes[i], sizes[i+1], activation))

    def forward(self, x: List[float]) -> List[float]:
        a = x[:]
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def loss_and_gradient(self, output: List[float], target: List[float]):
        if self.task == "classification":
            p = clamp(output[0], 1e-7, 1 - 1e-7)
            y = target[0]
            loss = -(y * math.log(p) + (1-y) * math.log(1-p))
            grad = [-(y/p) + (1-y)/(1-p)]
            return loss, grad

        error = output[0] - target[0]
        return 0.5 * error * error, [error]

    def train_sample(self, x: List[float], target: List[float], lr=0.03):
        out = self.forward(x)
        loss, grad = self.loss_and_gradient(out, target)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        for layer in self.layers:
            layer.update_sgd(lr)

        return out[0], loss

    def parameter_count(self) -> int:
        return sum(layer.in_size * layer.out_size + layer.out_size for layer in self.layers)

if __name__ == "__main__":
    random.seed(7)

    model = MLP([1, 4, 4, 1], hidden_activation="tanh", task="regression")
    print("Architecture:", model.sizes)
    print("Parameters:", model.parameter_count())

    data = []
    for i in range(80):
        x = -2.0 + 4.0 * i / 79.0
        y = 2.5*x + 1.0
        data.append(([x], [y]))

    for epoch in range(1, 301):
        total = 0.0
        random.shuffle(data)

        for x, y in data:
            _, loss = model.train_sample(x, y, lr=0.01)
            total += loss

        if epoch % 25 == 0:
            print(f"epoch={epoch:03d} loss={total/len(data):.8f}")

    for test_x in [-2, -1, 0, 1, 2]:
        pred = model.forward([float(test_x)])[0]
        print(f"x={test_x:>2} target={2.5*test_x+1:>6.2f} pred={pred:>9.5f}")
