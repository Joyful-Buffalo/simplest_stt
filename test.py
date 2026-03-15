import os
from pathlib import Path

import torch

from utils.JsonlASRDataset import JsonlASRDataset
from utils.asr_collate_fn import asr_collate_fn
from utils.CharTokenizer import CharTokenizer
from utils.dynamic_batch_sampler import DynamicBatchSampler
from utils.load_func import load_proj_root, load_yaml
from models.ctc_conformer import CTCConformer
from utils.decoder import ctc_greedy_decode_ids
from utils.char_edit_distance import char_edit_distance
from tqdm import tqdm

class Test:
    def __init__(self, proj_root=load_proj_root()):
        self.yaml_config_path = Path(f"{proj_root}/config/baseline.yaml")
        self.yaml_config = load_yaml(self.yaml_config_path)

        self.dataset_config = self.yaml_config["dataset_config"]
        self.model_config = self.yaml_config["model_config"]

        self.char_tokenizer = CharTokenizer(
            chars=None,
            jsonl_paths=[os.path.join(proj_root, v) for _ , v in self.yaml_config["jsonl_paths"].items()],
        )
        self.test_set = JsonlASRDataset(
            jsonl_file_path=os.path.join(proj_root, self.yaml_config['jsonl_paths']['test_jsonl']),
            tokenizer=self.char_tokenizer,
            config=self.dataset_config,
        )

        self.test_sampler = DynamicBatchSampler(
            lengths=self.test_set.num_frames,
            max_frame_per_batch=self.dataset_config["max_frame_per_batch"],
            shuffle=False,
            bucket_size=self.dataset_config["bucket_size"],
        ) 

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_set,
            batch_sampler=self.test_sampler,
            collate_fn=asr_collate_fn,
            num_workers=self.dataset_config["num_workers"],
            pin_memory=self.dataset_config["pin_memory"],
            prefetch_factor=self.dataset_config["prefetch_factor"],
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(os.path.join(proj_root, 'log', '20260315-180311', 'best_model.pth'))

    def load_model(self, model_path='None'):
        origin_weight = torch.load(model_path, map_location=self.device)
        weight = {}
        for k, v in origin_weight.items():
            if k.startswith('_orig_mod.'):
                weight[k[10:]] = v
            else:
                weight[k] = v
        model = CTCConformer(
            input_dim=self.yaml_config['dataset_config']['fbank_config']['feature_dim'],
            vocab_size=self.char_tokenizer.vocab_size(),
            **self.yaml_config['model_config']['encoder']
        )
        model.load_state_dict(weight)
        return model.to(self.device)
    
    def test(self):
        self.model.eval()
        cer_sum = 0
        total_chars = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, ncols=80, desc="Testing"):
                features = batch['features'].to(self.device)
                feature_lengths = batch['feature_lengths'].to(self.device)
                targets = batch['targets'].to(self.device)
                targets_lengths = batch['target_lengths'].to(self.device)

                logits, out_lens = self.model(features, feature_lengths)

                pred_ids_batch = ctc_greedy_decode_ids(
                    logit_batch=logits,
                    lengths=out_lens,
                    blank_id=0
                )
                offset = 0
                batch_cer_sum = 0
                batch_total_chars = 0
                for i in range(len(pred_ids_batch)):
                    cur_len = int(targets_lengths[i].item())
                    target_ids = targets[offset:offset+cur_len].tolist()
                    offset += cur_len
                    target_str = self.char_tokenizer.decode(target_ids)
                    pred_str = self.char_tokenizer.decode(pred_ids_batch[i])
                    cer = char_edit_distance(target_str, pred_str)
                    batch_cer_sum += cer
                    batch_total_chars += len(target_str)
                cer_sum += batch_cer_sum
                total_chars += batch_total_chars
        cer = cer_sum / total_chars if total_chars > 0 else 0
        print(f"Test CER: {cer * 100:.4f}%")


if __name__ == "__main__":
    tester = Test()
    tester.test()




