import json
from typing import List


class CharTokenizer:
    def __init__(self, chars: str=None, jsonl_paths:List[str]=None):
        self.jsonl_paths = jsonl_paths if jsonl_paths is not None else []
        self.chars = chars if chars is not None else ''
        self.chars = self.build_chars()
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}

    def build_chars(self):
        char_set = set()
        for jsonl in self.jsonl_paths:
            with open(jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    text = data['txt']
                    for char in text:
                        char_set.add(char)
        for char in self.chars:
            char_set.add(char)
        char_set = ''.join(sorted(list(char_set)))
        return char_set

    def encode(self, text: str):
        return [self.char2idx[char] for char in text if char in self.char2idx]

    def decode(self, indices: list):
        return ''.join([self.idx2char[idx] for idx in indices if idx in self.idx2char])