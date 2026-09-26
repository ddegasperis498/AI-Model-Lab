from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class BPETokenizerState:
    vocab: Dict[int, str]
    merges: List[Tuple[int, int, int]]


class TrainableByteBPETokenizer:
    """Small byte-level BPE tokenizer for educational/local experiments.

    IDs 0..255 are raw bytes. Learned merge tokens start at 256.
    The implementation is intentionally compact and deterministic; it is not
    intended to replace highly optimized production tokenizers.
    """

    def __init__(self):
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: List[Tuple[int, int, int]] = []

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _replace_pair(tokens: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        a, b = pair
        out: List[int] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def train(self, text: str, target_vocab_size: int = 384, min_pair_count: int = 2):
        if target_vocab_size < 256:
            raise ValueError('target_vocab_size deve essere >= 256')
        tokens = list(text.encode('utf-8', errors='replace'))
        if len(tokens) < 2:
            return

        while len(self.vocab) < target_vocab_size and len(tokens) >= 2:
            counts = Counter(zip(tokens, tokens[1:]))
            if not counts:
                break
            pair, count = counts.most_common(1)[0]
            if count < min_pair_count:
                break
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges.append((pair[0], pair[1], new_id))
            tokens = self._replace_pair(tokens, pair, new_id)

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode('utf-8', errors='replace'))
        for a, b, new_id in self.merges:
            tokens = self._replace_pair(tokens, (a, b), new_id)
        return tokens

    def decode(self, ids: Iterable[int]) -> str:
        raw = b''.join(self.vocab.get(int(i), b'?') for i in ids)
        return raw.decode('utf-8', errors='replace')

    def token_bytes(self, token_id: int) -> bytes:
        return self.vocab[int(token_id)]

    def state_dict(self) -> dict:
        return {
            'kind': 'byte_bpe',
            'vocab': {str(k): v.hex() for k, v in self.vocab.items()},
            'merges': [[a, b, new_id] for a, b, new_id in self.merges],
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> 'TrainableByteBPETokenizer':
        tok = cls()
        tok.vocab = {int(k): bytes.fromhex(v) for k, v in state['vocab'].items()}
        tok.merges = [tuple(map(int, row)) for row in state.get('merges', [])]
        return tok

    def preview(self, text: str, limit: int = 128) -> List[dict]:
        ids = self.encode(text)[:limit]
        rows = []
        for token_id in ids:
            b = self.vocab[token_id]
            rows.append({
                'id': token_id,
                'hex': b.hex(' '),
                'text': b.decode('utf-8', errors='replace'),
                'bytes': len(b),
            })
        return rows
