import json
import torch
from torch.utils.data import Dataset
import torchaudio

from CharTokenizer import CharTokenizer

class JsonlASRDataset(Dataset):
    def __init__(self, jsonl_file_path:str, tokenizer:CharTokenizer=None):
        super().__init__()
        self.jsonl_file_path = jsonl_file_path
        self.json_list = self.build_list()
        self.tokenizer = tokenizer

    def build_list(self):
        json_list = []
        with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                json_list.append(data)
        return json_list
    
    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx)->tuple[torch.Tensor, torch.Tensor]:
        item = self.json_list[idx]
        path = item['path']
        txt = item['txt']
        wav, *_ = torchaudio.load(path)
        fbank  = torchaudio.compliance.kaldi.fbank(wav)
        
        return fbank, torch.tensor(self.tokenizer.encode(txt),dtype=torch.long)
        