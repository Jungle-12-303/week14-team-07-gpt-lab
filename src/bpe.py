# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json
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
        #raise NotImplementedError("에러 메시지를 작성해주세요")
        self.vocab_size = vocab_size
        self._init_special_tokens()

    def _init_special_tokens(self):
        """
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        self.token_to_id.update(SPECIAL_IDS)
        self.token_to_id.update({bytes([byte_value]) : BYTE_OFFSET + byte_value for byte_value in range(0, NUM_BYTES)})

        self.id_to_token.update({idx : token for token, idx in self.token_to_id.items()})

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
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다. => 머지 룰..
        - `self.merges` => 튜플 한 쌍을 등록, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """

        self._init_special_tokens()
        
        token_ids = [BYTE_OFFSET + token for token in corpus.encode("utf-8")] # 8byte로 잘라서 배열 저장

        #--------------------------------
        def get_most_frequent_pair(token_ids: list[int]) -> tuple[int, int] | None:

            if len(token_ids) < 2:
                return None

            count_voc = {}
            # 전체를 순회한다.
            i = 0
            while(i < len(token_ids)-1):
                # 앞, 뒤를 무조건 한 쌍으로 간주하고 시작
                # 빈도를 카운트
                pair = (token_ids[i], token_ids[i+1])
                count_voc.update({pair : count_voc.get(pair,0)+1})

                i += 1
            
            return max(count_voc, key=lambda x: count_voc[x])
        
        def replace_pair(token_ids: list[int], pair: tuple[int, int]) -> list[int]:
            # 바꿔야 하는 위치를 찾아서 거기에 새 아이디를 넣어줘야 함.
            # 새 리스트... 점프하면 됨..
            temp_merge = []
            i = 0
            while(i < len(token_ids)-1):
                temp_left = token_ids[i]
                temp_right = token_ids[i+1]

                if(temp_left == pair[0] and temp_right == pair[1]):
                    merge_id = self.token_to_id.get((pair[0], pair[1]))
                    
                    if(merge_id == None):
                        raise ValueError("merge_id가 None...")
                    temp_merge.append(merge_id)

                    i += 2 # 두 칸 소비함
                else:
                    temp_merge.append(token_ids[i])

                    i += 1
            
            if(i == len(token_ids)-1): # bytes의 요소가 1개거나, 마지막 요소가 남았을 때
                temp_merge.append(token_ids[-1])

            return temp_merge
    

        #--------------------------------

        while(len(self.id_to_token) < self.vocab_size):
            pair = get_most_frequent_pair(token_ids) # tuple[int,int]
            if(pair is None):
                break

            new_token_id = len(self.id_to_token)
            
            self.token_to_id.update({pair : new_token_id})
            self.id_to_token.update({new_token_id : pair})
            self.merges.append(pair)
            token_ids = replace_pair(token_ids, pair)



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
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
            "id_to_token": {
                str(token_id): serialize_token(token)
                for token_id, token in self.id_to_token.items()
            },
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def load(self, path: str | Path):
        """
        save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        def deserialize_token(token_data):
            token_type = token_data["type"]
            token_value = token_data["value"]

            if token_type == "bytes":
                return bytes(token_value)
            if token_type == "tuple":
                return tuple(token_value)
            return token_value

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.vocab_size = data.get("vocab_size", self.vocab_size)
        self.id_to_token = {
            int(token_id): deserialize_token(token_data)
            for token_id, token_data in data["id_to_token"].items()
        }
        self.token_to_id = {
            token: token_id for token_id, token in self.id_to_token.items()
        }
        self.merges = [tuple(pair) for pair in data["merges"]]

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
        

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """

        #--------------------------------

        byte_valuse: list[int] = []
        text_pieces: list[str] = []

        # str로 진짜 디코딩
        def flush_bytes() -> None:
            text_pieces.append(bytes(byte_valuse).decode("utf-8", errors="replace"))
            byte_valuse.clear()
        
        # 재귀적으로 아이디를 토큰으로 변환
        def expand_to_bytes(token_id:int) -> list[int]:
            token = self.id_to_token[token_id]
            if isinstance(token, bytes):
                return list(token)
            if isinstance(token, tuple):
                return expand_to_bytes(token[0]) + expand_to_bytes(token[1]) 
            
            raise ValueError("fail..")


        #--------------------------------

        # 여기서 할꺼임

        # id리스트만큼 반복...

        # (65, 64) = 260
        # (32, 12) = 261
        # (260, 261) = 234

        for id in ids:
            token = self.id_to_token.get(int(id))
            if token is None:
                if skip_special == False:
                    flush_bytes()
                    text_pieces.append(UNK_TOKEN)
                continue

            if isinstance(token, str):
                if (skip_special == False) and (token in SPECIAL_TOKENS):
                    flush_bytes()
                    text_pieces.append(token)
                continue

            byte_valuse.extend(expand_to_bytes(int(id)))

        flush_bytes()
        return "".join(text_pieces)