# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from collections import Counter
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
        self._init_special_tokens()

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        # Initialize id_to_token
        idx = 0
        for token in SPECIAL_IDS: 
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
    
    def tutuple_to_bytes(self, t: tuple[tuple, ...]) -> bytes:
        return bytes(b for p in t for b in p)

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        breakpoint()
        byte_corpus = corpus.encode("utf-8")
        word_freq = Counter(byte_corpus.split())

        # 초기 vocab: 단어를 UTF-8 byte id tuple로 표현
        # 토큰 타입: tuple[int, ...]
        # ex: {((65,), (66)): 3, ...}
        words = {
            tuple((w,) for w in word): freq
            for word, freq in word_freq.items()
        }

        vocabulary = set()
        for word in words:
            vocabulary.update(word)

        while len(self.id_to_token) < self.vocab_size:
            # 빈도수 계산을 위한 토큰 쌍
            pairs = {}
            for k, v in words.items():
                for i in range(len(k)-1):
                    a, b = k[i], k[i+1]
                    pair = (a, b)
                    freq = v
                    if pair in pairs:
                        pairs[pair] += freq
                    else:
                        pairs[pair] = freq

            if not pairs:
                break

            pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
            # pair: 최대 빈도 토큰 쌍: tuple ((65,), (66,))
            pair = pairs[0][0]
            merged_token = pair[0] + pair[1]

            # merge
            new_words = {}
            for key, value in words.items():
                old_key = key
                new_key = []
                changed = False
                n = len(key)
                i = 0
                while i < n:
                    if i < n - 1:
                        comp_tuple = key[i:i+2]
                        if comp_tuple == pair:
                            changed = True
                            new_key.append(merged_token)
                            i += 1
                        else:
                            new_key.append(key[i])
                    else:
                        new_key.append(key[i])
                    i += 1
                new_key = tuple(new_key)
                if changed:
                    if new_key in new_words:
                        new_words[new_key] += value
                    else:
                        new_words[new_key] = value
                else:
                    if old_key in new_words:
                        new_words[old_key] += value
                    else:
                        new_words[old_key] = value

            self.merges.append(pair)
            vocabulary.add(merged_token)
            words = new_words

            idx = len(self.id_to_token)
            w = self.tutuple_to_bytes(pair)
            self.id_to_token[idx] = w
            self.token_to_id[w] = idx
            # idx += 1

        breakpoint()
        # update token_to_id
        # and id_to_token
        # idx = len(self.token_to_id)
        # # for k, v in self.merges.items():
        # for k in self.merges:
        #     w = self.tutuple_to_bytes(k)
        #     self.id_to_token[idx] = w
        #     self.token_to_id[w] = idx
        #     idx += 1
        # print(self.id_to_token)
        # print(self.token_to_id)
        # print(self.merges)
        # return words

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        bytes_texts = [x.encode('utf-8') for x in text.split()]
        # 이중 for문: merge rule > word list
        # merge rule
        for merge in self.merges:
            # word list
            for btext in bytes_texts:
                # word
                for b in btext:
                    pass
        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
