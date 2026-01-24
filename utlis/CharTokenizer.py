import json
from typing import List


class CharTokenizer:
    def __init__(self, chars: str=None, jsonls:List[str]=None):
        self.jsonls = jsonls
        self.chars = chars
        self.chars = self.build_chars()
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}

    def build_chars(self):
        char_set = set()
        for jsonl in self.jsonls:
            with open(jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    text = data['txt']
                    for char in text:
                        char_set.add(char)
        for char in self.chars:
            char_set.add(char)
        return char_set

    def encode(self, text: str):
        return [self.char2idx[char] for char in text if char in self.char2idx]

    def decode(self, indices: list):
        return ''.join([self.idx2char[idx] for idx in indices if idx in self.idx2char])