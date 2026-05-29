# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json


PAD_TOKEN = "<pad>" # 빈칸 채우기용(padding)
UNK_TOKEN = "<unk>" # 모르는 단어(unkonwn)
BOS_TOKEN = "<bos>" # 문장 시작(begging of sequence)
EOS_TOKEN = "<eos>" # 문장 끝(end of sequence)

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)   # 4
NUM_BYTES = 256 # 0~255까지


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
        self.id_to_token = {}   # decode: id → token (0 -> "<pad>")
        self.token_to_id = {}   # encode: token → id ("<pad>" -> 0)
        self.merges = []

    # 특수 토큰/byte 토큰 등록, token-id 양방향 매핑
    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        ex) bytes([65]) = b'A' 이렇게 나옴 근데 특수 토큰 있으니까 token_id는 69가 됨
        """

        """
        token ID 0~3은 특수 토큰이 쓰고 있으니까, byte token ID는 4를 더해서 쓴다.
        byte:     [65, 236, 149, 136, 235, 133, 149]
        token ID: [69, 240, 153, 140, 239, 137, 153] 이런식으로 4씩 더함
        print(bytes([65]))              # b'A'
        print(self.id_to_token[69])     # b'A'
        print(self.token_to_id[b'A'])   # 69
        """

        # 초기화(방어 장치)
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        for token in SPECIAL_TOKENS:
            token_id = SPECIAL_IDS[token]
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        for b in range(NUM_BYTES):
            token_id = BYTE_OFFSET + b
            token = bytes([b])
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

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

    # tokenizer 규칙을 만든다
    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.  vocab = tokeizer가 사용하는 토큰 사전

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다. corpus = 학습용 텍스트 데이터 전체
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        self._init_special_tokens()

        ids = []
        # corpus를 byte token ID 리스트로 바꾸기
        for byte_value in corpus.encode("utf-8"):
            token = bytes([byte_value])
            token_id = self.token_to_id[token]
            ids.append(token_id)

        # vocab 크기가 목표치에 도달할 때까지 반복
        while len(self.id_to_token) < self.vocab_size:
            pair_counts = {}    # 예) (69, 70): 2

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)

            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            self.merges.append(best_pair)

            # 기존 ids 안에서 best_pair 두 개를 새 token ID로 바꿈
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            ids = new_ids

    # 만든 규칙을 파일로 저장한다
    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        serialized_vocab = []

        # JSON은 bytes와 tuple을 그대로 몰라서 타입도 같이 저장함
        for token_id, token in self.id_to_token.items():
            if isinstance(token, str):
                token_data = {"type": "str", "value": token}
            elif isinstance(token, bytes):
                token_data = {"type": "bytes", "value": list(token)}
            elif isinstance(token, tuple):
                token_data = {"type": "tuple", "value": list(token)}
            else:
                raise TypeError(f"Unsupported token type: {type(token)}")

            serialized_vocab.append({
                "id": token_id,
                "token": token_data,
            })

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": serialized_vocab,
            "merges": [list(pair) for pair in self.merges],
        }

        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 저장된 규칙을 다시 메모리로 복원한다
    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}

        for item in data["id_to_token"]:
            token_id = int(item["id"])
            token_data = item["token"]
            token_type = token_data["type"]
            value = token_data["value"]

            if token_type == "str":
                token = value
            elif token_type == "bytes":
                token = bytes(value)
            elif token_type == "tuple":
                token = tuple(value)
            else:
                raise ValueError(f"Unsupported token type: {token_type}")

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        # JSON에서는 튜플이 리스트로 저장되기 때문에 다시 튜플로 저장
        self.merges = [tuple(pair) for pair in data["merges"]]

    # add_bos_eos: bool = False: 기본값은 False, 필요하면 문장 앞뒤에 <bos>, <eos>를 붙임
    # -> list[int]: 반환값은 정수 리스트
    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - [O] 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - [ ] train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - [O] add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        if not self.token_to_id:
            self._init_special_tokens()
        byte_values = text.encode("utf-8")
        ids = []    # 최종 token ID들을 담을 빈 리스트

        for v in byte_values:
            token = bytes([v])  # self.token_to_id의 key가 bytes 형태니까 다시 bytes 타입으로 바꿈
            token_id = self.token_to_id[token]  #현재 byte token에 해당하는 ID를 찾기
            ids.append(token_id)
        
        # merge 규칙을 학습 순서대로 하나씩 적용
        for pair in self.merges:
            if pair not in self.token_to_id:
                continue

            new_id = self.token_to_id[pair]
            new_ids = []
            i = 0

            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            ids = new_ids

        # 필요하면 앞뒤에 <bos>,<eos> id 붙임
        if add_bos_eos == True:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]
            
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.
        주의:
        - [O] merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - [O] byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        if not self.id_to_token:
            self._init_special_tokens()

        # 튜플 형태로 받은 token ID를 펼쳐서 최종 byte 숫자 리스트로 만들기
        def expand_token(token_id: int) -> list[int]:
            token = self.id_to_token[token_id]

            if isinstance(token, bytes):
                return [token[0]]

            if isinstance(token, tuple):
                left_id, right_id = token
                return expand_token(left_id) + expand_token(right_id)

            return []
    
        byte_list = []

        for token_id in ids:
            if skip_special == True and token_id in SPECIAL_IDS.values():
                continue
            token = self.id_to_token[token_id]
            # if type(token) == bytes:
            if isinstance(token, bytes):    # isinstance(객체, 타입) - 이 객체가 이 타입이 맞냐? (T/F 반환)
                byte_list.append(token[0])
            
        return bytes(byte_list).decode('utf-8')
        