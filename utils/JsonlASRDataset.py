import json
import torch
from torch.utils.data import Dataset
import torchaudio

from utils.CharTokenizer import CharTokenizer

class JsonlASRDataset(Dataset):
    def __init__(self, jsonl_file_path:str, config, tokenizer:CharTokenizer=None):
        super().__init__()
        self.fbank_config = config['fbank_config']
        self.frame_length_ms = self.fbank_config['frame_length_ms']
        self.frame_shift_ms = self.fbank_config['frame_shift_ms']
        self.tokenizer = tokenizer
        self.jsonl_file_path = jsonl_file_path
        self.num_frames = []
        self.json_list = self.build_list()

    def build_list(self):
        json_list = []
        with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                json_list.append(data)
                duration = data['duration']
                num_frame = int((duration*1000 - self.frame_length_ms) / self.frame_shift_ms) + 1
                self.num_frames.append(num_frame)
        return json_list
    
    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx)->tuple[torch.Tensor, torch.Tensor]:
        item = self.json_list[idx]
        path = item['path']
        txt = item['txt']
        wav, sample_rate = torchaudio.load(path)
        expected_rate = self.fbank_config['sample_rate']
        if sample_rate != expected_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, expected_rate)
            sample_rate = expected_rate
        fbank = torchaudio.compliance.kaldi.fbank(
            wav,
            sample_frequency=sample_rate,
            num_mel_bins=self.fbank_config['feature_dim'],
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
            dither=self.fbank_config['dither'],
        )
        
        return fbank, torch.tensor(self.tokenizer.encode(txt),dtype=torch.long)
        
