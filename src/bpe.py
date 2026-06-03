# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE tokenizer.
"""

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
    UTF-8 byte-level BPE tokenizer.

    Recommended ID layout:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: raw bytes 0~255
    - 260+: tokens created by BPE merges
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self._init_special_tokens()

    def _init_special_tokens(self):
        """
        1. Register 4 special tokens at fixed IDs 0~3.
        2. Register bytes 0~255 as bytes([byte_value]) at IDs 4~259.
        """
        for i, token in enumerate(SPECIAL_TOKENS):
            self.id_to_token[i] = token
            self.token_to_id[token] = i

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

    def train(self, corpus: str):
        """
        Learn BPE merge rules and vocabulary from the corpus.
        """
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self._init_special_tokens()

        sequence = [BYTE_OFFSET + b for b in corpus.encode("utf-8")]

        while len(self.id_to_token) < self.vocab_size and len(sequence) >= 2:
            token_pair_count = {}
            max_pair_count = 0
            max_pair = None

            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                if pair in token_pair_count:
                    token_pair_count[pair] += 1
                else:
                    token_pair_count[pair] = 1

                if token_pair_count[pair] > max_pair_count:
                    max_pair_count = token_pair_count[pair]
                    max_pair = pair

            if max_pair is None:
                break
            if max_pair_count < 2:
                break

            new_id = len(self.id_to_token)
            if new_id >= self.vocab_size:
                break

            self.merges.append(max_pair)
            self.id_to_token[new_id] = max_pair
            self.token_to_id[max_pair] = new_id

            new_sequence = []
            i = 0
            while i < len(sequence):
                if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == max_pair:
                    new_sequence.append(new_id)
                    i += 2
                else:
                    new_sequence.append(sequence[i])
                    i += 1

            sequence = new_sequence

    def save(self, path: str | Path):
        """
        Save vocabulary and merge rules to JSON.
        """
        save_data = {}
        vocabulary = []

        for token_id, token in self.id_to_token.items():
            if 0 <= token_id <= 3:
                vocabulary.append([token_id, "special", token])
            elif 4 <= token_id <= 259:
                vocabulary.append([token_id, "byte", token[0]])
            else:
                vocabulary.append([token_id, "merge", list(token)])

        save_data["vocabulary"] = vocabulary
        save_data["vocab_size"] = self.vocab_size
        save_data["merges"] = [list(pair) for pair in self.merges]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f)

    def load(self, path: str | Path):
        """
        Restore vocabulary and merge rules from a JSON file saved by save().
        """
        with open(path, "r", encoding="utf-8") as f:
            load_data = json.load(f)

        self.vocab_size = load_data["vocab_size"]

        restore_merges = []
        for left, right in load_data["merges"]:
            restore_merges.append((left, right))

        self.merges = restore_merges
        self.id_to_token = {}
        self.token_to_id = {}

        for token_id, token_type, token in load_data["vocabulary"]:
            if token_type == "byte":
                token = bytes([token])
            elif token_type == "merge":
                left, right = token
                token = (left, right)

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        Convert text to a token ID list.
        """
        text_bytes = text.encode("utf-8")
        result = [self.token_to_id[bytes([b])] for b in text_bytes]

        current_sequence = result
        for pair in self.merges:
            new_id = self.token_to_id[pair]
            merged_sequence = []
            i = 0

            while i < len(current_sequence):
                if i + 1 < len(current_sequence) and (current_sequence[i], current_sequence[i + 1]) == pair:
                    merged_sequence.append(new_id)
                    i += 2
                else:
                    merged_sequence.append(current_sequence[i])
                    i += 1

            current_sequence = merged_sequence

        result = current_sequence

        if add_bos_eos:
            result.insert(0, self.get_bos_id())
            result.append(self.get_eos_id())

        return result

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        Restore text from a token ID list.
        """

        def expand_to_bytes(token_id: int) -> list[int]:
            expanded = []
            stack = [token_id]

            while stack:
                current_id = stack.pop()
                token = self.id_to_token[current_id]

                if isinstance(token, bytes):
                    expanded.append(token[0])
                elif isinstance(token, tuple):
                    left, right = token
                    stack.append(right)
                    stack.append(left)
                elif isinstance(token, str):
                    expanded.extend(token.encode("utf-8", errors="replace"))

            return expanded

        byte_list = []
        for token_id in ids:
            if 0 <= token_id <= 3 and skip_special:
                continue
            byte_list.extend(expand_to_bytes(token_id))

        return bytes(byte_list).decode("utf-8", errors="replace")
