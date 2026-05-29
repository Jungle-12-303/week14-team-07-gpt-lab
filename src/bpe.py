# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
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

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        for i in range(len(SPECIAL_TOKENS)):
            self.id_to_token[i] = SPECIAL_TOKENS[i]
            self.token_to_id[SPECIAL_TOKENS[i]] = i
        
        for i in range(4, 260):
            self.id_to_token[i] = bytes([i-4])
            self.token_to_id[bytes([i-4])] = i

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
    TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

    구현 힌트:
    - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
    - 가장 자주 등장하는 이웃 token pair를 찾습니다.
    - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
    - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
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
            left = sequence[i]
            right = sequence[i + 1]
            pair = (left, right)

            if pair in token_pair_count:
                token_pair_count[pair] += 1
            else:
                token_pair_count[pair] = 1

            if token_pair_count[pair] > max_pair_count:
                max_pair_count = token_pair_count[pair]
                max_pair = pair

        if max_pair is None:
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
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        save_data = {}
        vocabulary = []
        
        for id, token in self.id_to_token.items():
            if (0 <= id and id <=3):
                vocabulary.append([id, "special", token])
                
            if (4<= id <=259):
                vocabulary.append([id, "byte", token[0]])

            if (id >= 260):
                vocabulary.append([id, "merge", token])
            
        save_data["vocabulary"] = vocabulary        
        save_data["vocab_size"] = self.vocab_size
        save_data["merges"] = self.merges

        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
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
            if (token_type == "byte"):
                token = bytes([token])

            if (token_type == "merge"):
                left, right = token
                token = (left, right)

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id


    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        text_list = list(bytes(text, "utf-8"))
        result = []
        for i in range(len(text_list)):
            result.append(self.token_to_id[bytes([text_list[i]])])

        # 추후 merge rule 적용 구현
        if (add_bos_eos):
            result.insert(0,self.get_bos_id())
            result.append(self.get_eos_id())
        
        return result
        
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        byte_list = []

        for token_id in ids:
            if (0 <= token_id and token_id <=3 and skip_special == True):
                continue
                
            # if (token_id >= 260):
            #     for i in range(len(self.merges)):
            #         # 추후 merge rule 적용 구현
            #         pass
            
            if (4 <= token_id <= 259):
                byte_list.append(token_id-4)
            
        return bytes(byte_list).decode("utf-8")
            
