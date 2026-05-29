# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import re


PAD_TOKEN = "<pad>" # Padding Token 문장 길이 맞추기 위한 패딩
UNK_TOKEN = "<unk>" # Unknown Token 모르는 단어 대체
BOS_TOKEN = "<bos>" # Beginning of Sequence 새로운 문장 표식
EOS_TOKEN = "<eos>" # End of Sequence 문장 끝 표식

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
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        self.token_to_id.update(SPECIAL_IDS)
        self.token_to_id.update({bytes([byte_value]) : BYTE_OFFSET+byte_value for byte_value in range(0, NUM_BYTES)})

        self.id_to_token.update({idx : token for token, idx in self.token_to_id.items()})

        # raise NotImplementedError("_init_special_tokens를 구현하세요.")

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
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다. => 머지 룰..
        - `self.merges` => 튜플 한 쌍을 등록, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        UTF8_bytes = [token for token in corpus.encode("utf-8")] # 8byte로 잘라서 배열 저장

        UTF8_byte_ID = []
        for token in UTF8_bytes:
            idx = self.token_to_id.get(bytes([token])) # 토큰을 바이트로 전환해서 id 조회
            UTF8_byte_ID.append(idx)



        raise NotImplementedError("BPETokenizer.train을 구현하세요.")

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
        문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        # text = "Fly me to the moon"
        UTF8_bytes = [token for token in text.encode("utf-8")] # 8byte로 잘라서 배열 저장

        UTF8_byte_ID = []
        
        for token in UTF8_bytes:
            idx = self.token_to_id.get(bytes([token])) # 토큰을 바이트로 전환해서 id 조회
            UTF8_byte_ID.append(idx)

        # merges 안에는 튜플 형태의 pair 목록이 있음.
        # [(a, b), (c, d), (e, f), ... ]
        for left_id, right_id in self.merges:
            temp_merge = []
            i = 0
            while(i < len(UTF8_byte_ID)-1):
                temp_left = UTF8_byte_ID[i]
                temp_right = UTF8_byte_ID[i+1]

                if(temp_left == left_id and temp_right == right_id):
                    merge_id = self.token_to_id.get((left_id, right_id))
                    
                    if(merge_id == None):
                        raise ValueError("merge_id가 None...")
                    temp_merge.append(merge_id)

                    i += 2 # 두 칸 소비함
                else:
                    temp_merge.append(UTF8_byte_ID[i])

                    i += 1
            
            if(i == len(UTF8_byte_ID)-1): # UTF8_byte_ID의 요소가 1개거나, 마지막 요소가 남았을 때
                temp_merge.append(UTF8_byte_ID[-1])

            UTF8_byte_ID = temp_merge


        if(add_bos_eos == True): # 문장 처음/끝 표식 붙이기
            UTF8_byte_ID.insert(0, self.get_bos_id())
            UTF8_byte_ID.append(self.get_eos_id())

        return UTF8_byte_ID
        
        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
