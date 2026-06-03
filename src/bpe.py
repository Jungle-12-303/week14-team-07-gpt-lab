# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from collections import Counter, defaultdict
from pathlib import Path
import heapq
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
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self._init_special_tokens()

    def _init_special_tokens(self):
        """
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        # Initialize id_to_token
        idx = 0
        for token in SPECIAL_TOKENS:
            self.id_to_token[idx] = token
            idx += 1
        for b in range(0, NUM_BYTES):
            self.id_to_token[idx] = bytes([b])
            idx += 1
        
        # Initialize token_to_id
        for k, v in dict.items(self.id_to_token):
            self.token_to_id[v] = k


    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]
    
    def train(self, corpus: str):
        """
        코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        tokens = [BYTE_OFFSET + b for b in corpus.encode("utf-8")]
        if len(tokens) < 2:
            return

        prev = [i - 1 for i in range(len(tokens))]
        next_ = [i + 1 for i in range(len(tokens))]
        next_[-1] = -1

        pair_counts = Counter()
        pair_positions = defaultdict(set)
        pair_position_heaps = defaultdict(list)
        heap = []

        def pair_at(index):
            if index == -1 or tokens[index] is None:
                return None
            right = next_[index]
            if right == -1 or tokens[right] is None:
                return None
            return (tokens[index], tokens[right])

        def push_pair(pair):
            count = pair_counts[pair]
            if count > 0:
                first_pos = first_position(pair)
                if first_pos is not None:
                    heapq.heappush(heap, (-count, first_pos, pair))

        def first_position(pair):
            positions = pair_position_heaps[pair]
            active_positions = pair_positions[pair]
            while positions and positions[0] not in active_positions:
                heapq.heappop(positions)
            if not positions:
                return None
            return positions[0]

        def add_pair(index):
            pair = pair_at(index)
            if pair is None:
                return
            pair_counts[pair] += 1
            pair_positions[pair].add(index)
            heapq.heappush(pair_position_heaps[pair], index)
            push_pair(pair)

        def remove_pair(index):
            pair = pair_at(index)
            if pair is None or index not in pair_positions[pair]:
                return
            pair_positions[pair].remove(index)
            pair_counts[pair] -= 1
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            else:
                push_pair(pair)

        for i in range(len(tokens) - 1):
            add_pair(i)

        while len(self.id_to_token) < self.vocab_size and heap:
            while heap:
                neg_count, first_pos, pair = heapq.heappop(heap)
                if (
                    pair_counts.get(pair, 0) == -neg_count
                    and pair_positions[pair]
                    and first_position(pair) == first_pos
                ):
                    break
            else:
                break

            if -neg_count <= 1:
                break

            merged_token = len(self.id_to_token)

            for left in sorted(pair_positions[pair]):
                right = next_[left]
                if right == -1 or pair_at(left) != pair:
                    continue

                before = prev[left]
                after = next_[right]

                remove_pair(before)
                remove_pair(left)
                remove_pair(right)

                tokens[left] = merged_token
                tokens[right] = None
                next_[left] = after
                if after != -1:
                    prev[after] = left

                add_pair(before)
                add_pair(left)

            self.merges.append(pair)
            self.id_to_token[merged_token] = pair
            self.token_to_id[pair] = merged_token

    def save(self, path: str | Path):
        """
        vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        def serialize_token(token):
            if isinstance(token, bytes):
                return {"type": "bytes", "value": list(token)}
            if isinstance(token, tuple):
                return {"type": "tuple", "value": list(token)}
            return {"type": "str", "value": token}

        data = {
            "merges": [list(pair) for pair in self.merges],
            "vocab_size": self.vocab_size,
            "id_to_token": {k: serialize_token(v) for k, v in self.id_to_token.items()}
        }
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f)

    def load(self, path: str | Path):
        """
        save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
        self.merges = [tuple(pair) for pair in data['merges']]
        self.vocab_size = data['vocab_size']

        def deserialize_token(token):
            if isinstance(token, dict):
                if token["type"] == "bytes":
                    return bytes(token["value"])
                if token["type"] == "tuple":
                    return tuple(token["value"])
                return token["value"]
            return token

        self.id_to_token = {
            int(k): deserialize_token(v)
            for k, v in data['id_to_token'].items()
        }
        self.token_to_id.clear()
        for k, v in self.id_to_token.items():
            self.token_to_id[v] = k

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        tokens = [BYTE_OFFSET + v for v in text.encode('utf-8')]
        for merge in self.merges:
            index = 0
            new_tokens = []
            while index < len(tokens):
                if index < len(tokens) - 1 and (tokens[index], tokens[index + 1]) == merge:
                    new_tokens.append(self.token_to_id[merge])
                    index += 2
                else:
                    new_tokens.append(tokens[index])
                    index += 1
            tokens = new_tokens
        ret = tokens
        if add_bos_eos:
            ret.insert(0, SPECIAL_IDS[BOS_TOKEN])
            ret.append(SPECIAL_IDS[EOS_TOKEN])
            return ret
        else:
            return ret

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        def token_to_bytes(token_id):
            token = self.id_to_token[token_id]
            if isinstance(token, bytes):
                return token
            if isinstance(token, tuple):
                return b''.join(token_to_bytes(i) for i in token)
            return b''

        if skip_special:
            byte_chunks = [
                token_to_bytes(i)
                for i in ids
                if self.id_to_token[i] not in SPECIAL_IDS
            ]
            result = b''.join(byte_chunks).decode('utf-8', errors='replace')
        else:
            # skip_special == False, 토큰을 그대로 출력한다.
            result = ''
            for i in ids:
                token = self.id_to_token[i]
                if token in SPECIAL_IDS:
                    result += token
                else:
                    result += token_to_bytes(i).decode('utf-8', errors='replace')
        return result
