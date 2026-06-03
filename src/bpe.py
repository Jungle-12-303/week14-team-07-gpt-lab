# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 항상 `text.encode("utf-8")` 기반 byte ID
시퀀스에서 merge를 학습하고 적용합니다.
"""

from collections import Counter
import json
from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token: dict[int, str | bytes | tuple[int, int]] = {}
        self.token_to_id: dict[str | bytes | tuple[int, int], int] = {}
        self.merges: list[tuple[int, int]] = []
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        for token, token_id in SPECIAL_IDS.items():
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            token = bytes([byte_value])
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def get_pad_id(self):
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        return SPECIAL_IDS[EOS_TOKEN]

    @staticmethod
    def _replace_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def train(self, corpus: str):
        self._init_special_tokens()
        ids = [BYTE_OFFSET + byte_value for byte_value in corpus.encode("utf-8")]

        while len(self.id_to_token) < self.vocab_size and len(ids) >= 2:
            pair_counts = Counter(zip(ids, ids[1:]))
            if not pair_counts:
                break

            best_pair, _ = pair_counts.most_common(1)[0]
            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            self.merges.append(best_pair)
            ids = self._replace_pair(ids, best_pair, new_id)

    @staticmethod
    def _serialize_token(token: str | bytes | tuple[int, int]) -> dict:
        if isinstance(token, bytes):
            return {"type": "bytes", "value": list(token)}
        if isinstance(token, tuple):
            return {"type": "tuple", "value": list(token)}
        return {"type": "str", "value": token}

    @staticmethod
    def _deserialize_token(data: dict) -> str | bytes | tuple[int, int]:
        token_type = data["type"]
        value = data["value"]
        if token_type == "bytes":
            return bytes(value)
        if token_type == "tuple":
            return tuple(value)
        if token_type == "str":
            return value
        raise ValueError(f"Unsupported token type: {token_type}")

    def save(self, path: str | Path):
        data = {
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
            "id_to_token": {
                str(token_id): self._serialize_token(token)
                for token_id, token in self.id_to_token.items()
            },
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab_size = data.get("vocab_size", self.vocab_size)
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.id_to_token = {
            int(token_id): self._deserialize_token(token_data)
            for token_id, token_data in data["id_to_token"].items()
        }
        self.token_to_id = {
            token: token_id for token_id, token in self.id_to_token.items()
        }

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        if not self.token_to_id:
            self._init_special_tokens()

        ids = [BYTE_OFFSET + byte_value for byte_value in text.encode("utf-8")]
        for pair in self.merges:
            new_id = self.token_to_id.get(pair)
            if new_id is not None:
                ids = self._replace_pair(ids, pair, new_id)

        if add_bos_eos:
            return [self.get_bos_id(), *ids, self.get_eos_id()]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if not self.id_to_token:
            self._init_special_tokens()

        def expand(token_id: int) -> bytes:
            token = self.id_to_token.get(token_id)
            if token is None:
                return b""
            if isinstance(token, bytes):
                return token
            if isinstance(token, tuple):
                return expand(token[0]) + expand(token[1])
            if skip_special and token in SPECIAL_IDS:
                return b""
            return token.encode("utf-8")

        byte_stream = b"".join(expand(token_id) for token_id in ids)
        return byte_stream.decode("utf-8", errors="replace")
