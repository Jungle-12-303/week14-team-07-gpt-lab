# -*- coding: utf-8 -*-
"""UTF-8 byte-level BPE tokenizer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """UTF-8 byte-level BPE tokenizer."""

    def __init__(self, vocab_size: int = 3000, min_frequency: int = 1):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        self.id_to_token: dict[int, str | bytes | tuple[int, int]] = {}
        self.token_to_id: dict[str | bytes | tuple[int, int], int] = {}
        self.merges: list[tuple[int, int]] = []

        self.token_to_id.update(SPECIAL_IDS)
        self.id_to_token.update({idx: token for token, idx in SPECIAL_IDS.items()})

        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            token = bytes([byte_value])
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def get_pad_id(self) -> int:
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self) -> int:
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self) -> int:
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self) -> int:
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        self._init_special_tokens()
        token_ids = [BYTE_OFFSET + byte for byte in corpus.encode("utf-8")]

        while len(self.id_to_token) < self.vocab_size:
            pair_counts = Counter(zip(token_ids, token_ids[1:]))
            if not pair_counts:
                break
            pair, freq = pair_counts.most_common(1)[0]
            if freq < self.min_frequency:
                break

            new_token_id = len(self.id_to_token)
            self.token_to_id[pair] = new_token_id
            self.id_to_token[new_token_id] = pair
            self.merges.append(pair)
            token_ids = self._replace_pair(token_ids, pair, new_token_id)

    def _replace_pair(
        self,
        token_ids: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:
        merged: list[int] = []
        index = 0
        while index < len(token_ids):
            if (
                index < len(token_ids) - 1
                and token_ids[index] == pair[0]
                and token_ids[index + 1] == pair[1]
            ):
                merged.append(new_token_id)
                index += 2
            else:
                merged.append(token_ids[index])
                index += 1
        return merged

    def save(self, path: str | Path):
        def serialize_token(token_value):
            if isinstance(token_value, bytes):
                return {"type": "bytes", "value": list(token_value)}
            if isinstance(token_value, tuple):
                return {"type": "tuple", "value": list(token_value)}
            return {"type": "str", "value": token_value}

        serialized = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "merges": [list(pair) for pair in self.merges],
            "id_to_token": {
                str(token_id): serialize_token(token)
                for token_id, token in self.id_to_token.items()
            },
        }
        Path(path).write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path):
        def deserialize_token(serialized_token):
            token_type = serialized_token["type"]
            token_value = serialized_token["value"]
            if token_type == "bytes":
                return bytes(token_value)
            if token_type == "tuple":
                return tuple(token_value)
            return token_value

        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab_size = saved.get("vocab_size", self.vocab_size)
        self.min_frequency = saved.get("min_frequency", self.min_frequency)
        self.id_to_token = {
            int(token_id): deserialize_token(token)
            for token_id, token in saved["id_to_token"].items()
        }
        self.token_to_id = {token: token_id for token_id, token in self.id_to_token.items()}
        self.merges = [tuple(pair) for pair in saved["merges"]]

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        token_ids = [BYTE_OFFSET + byte for byte in text.encode("utf-8")]
        for pair in self.merges:
            merged_id = self.token_to_id[pair]
            token_ids = self._replace_pair(token_ids, pair, merged_id)

        if add_bos_eos:
            return [self.get_bos_id(), *token_ids, self.get_eos_id()]
        return token_ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        byte_values: list[int] = []
        text_parts: list[str] = []

        def flush_byte_values():
            if byte_values:
                text_parts.append(bytes(byte_values).decode("utf-8", errors="replace"))
                byte_values.clear()

        def expand_token(token_id: int) -> list[int]:
            token = self.id_to_token[token_id]
            if isinstance(token, bytes):
                return list(token)
            if isinstance(token, tuple):
                return expand_token(token[0]) + expand_token(token[1])
            raise ValueError(f"Token id {token_id} is not a byte-bearing token.")

        for token_id in ids:
            token = self.id_to_token.get(int(token_id))
            if token is None:
                if not skip_special:
                    flush_byte_values()
                    text_parts.append(UNK_TOKEN)
                continue

            if isinstance(token, str):
                if not skip_special and token in SPECIAL_TOKENS:
                    flush_byte_values()
                    text_parts.append(token)
                continue

            byte_values.extend(expand_token(int(token_id)))

        flush_byte_values()
        return "".join(text_parts)
