import os
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.ctc_conformer import CTCConformer
from utils.JsonlASRDataset import JsonlASRDataset
from utils.load_func import load_proj_root, load_yaml
from utils.CharTokenizer import CharTokenizer
from utils.dynamic_batch_sampler import DynamicBatchSampler
from utils.asr_collate_fn import asr_collate_fn
from utils.noamLR import NoamLR
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn import functional
from utils.decoder import ctc_greedy_decode_ids
from utils.char_edit_distance import char_edit_distance

class Trainer:
    def __init__(self, proj_root=load_proj_root()):
        self.log_dir = Path(f"{proj_root}/log")/time.strftime("%Y%m%d-%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.yaml_config_path = Path(f"{proj_root}/config/baseline.yaml")
        self.yaml_config = load_yaml(self.yaml_config_path)
        self.dataset_config = self.yaml_config['dataset_config']
        self.model_config = self.yaml_config['model_config']

        self.char_tokenizer = CharTokenizer(
            chars=None, 
            jsonl_paths=[os.path.join(proj_root, v) for _,v in self.yaml_config["jsonl_paths"].items()]
        )
        print(f"Number of unique characters: {len(self.char_tokenizer.chars)}")
        self.train_set = JsonlASRDataset(
            jsonl_file_path=os.path.join(proj_root, self.yaml_config["jsonl_paths"]["train_jsonl"]),
            tokenizer=self.char_tokenizer,
            config=self.dataset_config
        )
        self.dev_set = JsonlASRDataset(
            jsonl_file_path=os.path.join(proj_root, self.yaml_config["jsonl_paths"]["dev_jsonl"]),
            tokenizer=self.char_tokenizer,
            config=self.dataset_config
        )

        self.train_sampler = DynamicBatchSampler(
            lengths=self.train_set.num_frames,
            max_frame_per_batch=self.dataset_config["max_frame_per_batch"],
            shuffle=self.dataset_config['shuffle'],
            bucket_size=self.dataset_config['bucket_size']
        )
        self.dev_sampler = DynamicBatchSampler(
            lengths=self.dev_set.num_frames,
            max_frame_per_batch=self.dataset_config["max_frame_per_batch"],
            shuffle=False,
            bucket_size=self.dataset_config['bucket_size']
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set,
            batch_sampler=self.train_sampler,
            collate_fn=asr_collate_fn,
            num_workers=self.dataset_config['num_workers'],
            pin_memory=self.dataset_config['pin_memory'],
            prefetch_factor=self.dataset_config['prefetch_factor']
        )
        self.dev_loader = torch.utils.data.DataLoader(
            dataset=self.dev_set,
            batch_sampler=self.dev_sampler,
            collate_fn=asr_collate_fn,
            num_workers=self.dataset_config['num_workers'],
            pin_memory=self.dataset_config['pin_memory'],
            prefetch_factor=self.dataset_config['prefetch_factor']
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = CTCConformer(
            input_dim=self.yaml_config['dataset_config']['fbank_config']['feature_dim'],
            vocab_size=self.char_tokenizer.vocab_size(),
            **self.model_config['encoder']
        ).to(self.device)
        self.loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction='sum')

        self.amp = torch.amp.autocast(enabled=self.yaml_config.get("use_amp", False), dtype=torch.bfloat16, device_type=self.device.type)
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), 
            lr=self.yaml_config["lr"], 
            weight_decay=self.yaml_config["weight_decay"],
            fused=self.yaml_config['fused']
        )
        if self.yaml_config.get("tf32", False) and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 for matmul and cudnn.")
        self.model = self.dynamic_pre_compile()
        self.warmup_lr = NoamLR(self.optimizer, 12000)

    def dynamic_pre_compile(self):
        warmup_batch = next(iter(self.train_loader))
        features = warmup_batch['features'].to(self.device)
        feature_lengths = warmup_batch['feature_lengths'].to(self.device)
        torch._dynamo.maybe_mark_dynamic(features, 0)
        torch._dynamo.maybe_mark_dynamic(features, 1)
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(self.model)
        f = open(self.log_dir / "dynamo_log.txt", "w")
        # print(torch._dynamo.explain(model)(features, feature_lengths), file=f)
        with torch.no_grad():
            _ = model(features, feature_lengths)
        return model
    

    def train_epoch(self):
        best_cer = float('inf')
        total_loss = 0.0
        total_batches = 0
        cer_sum = 0
        total_chars = 0
        self.model.train()
        with sdpa_kernel(SDPBackend.MATH):
            for batch_idx, batch in enumerate(tqdm(self.train_loader, ncols=80, desc="Training")):
                features = batch['features'].to(self.device)
                feature_lengths = batch['feature_lengths'].to(self.device)
                targets = batch['targets'].to(self.device)
                target_lengths = batch['target_lengths'].to(self.device)

                with self.amp:
                    logits, out_lens = self.model(features, feature_lengths)
                    log_probs = functional.log_softmax(logits, dim=-1).transpose(0, 1)  # (time, batch, vocab_size)
                loss = self.loss(log_probs, targets, out_lens, target_lengths)
                loss = loss / features.size(0)
                if batch_idx % 100 == 0:
                    invalid = (out_lens < target_lengths).sum().item()
                    # print("invalid_ratio", invalid / out_lens.numel())
                    # print(f'out_lens: {out_lens}, target_lengths: {target_lengths}')
                    with torch.no_grad():
                        pred_ids_batch = ctc_greedy_decode_ids(
                            logit_batch=logits,
                            lengths=out_lens,
                            blank_id=0
                        )
                        offset = 0
                        batch_cer_sum = 0
                        batch_total_chars = 0
                        for b in range(len(pred_ids_batch)):
                            cur_len = int(target_lengths[b].item())
                            target_ids = targets[offset:offset+cur_len].tolist()
                            offset += cur_len
                            target_str = self.char_tokenizer.decode(target_ids)
                            pred_str = self.char_tokenizer.decode(pred_ids_batch[b])
                            cer = char_edit_distance(target_str, pred_str)
                            batch_cer_sum += cer
                            batch_total_chars += len(target_str)
                        cer_sum += batch_cer_sum
                        total_chars += batch_total_chars
                        batch_cer_percent = (
                            (batch_cer_sum / batch_total_chars) * 100
                            if batch_total_chars > 0 else 0.0
                        )
                        if batch_cer_percent < best_cer:
                            best_cer = batch_cer_percent
                            torch.save(self.model.state_dict(), self.log_dir / "best_model.pth")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                self.warmup_lr.step()
                total_loss += loss.item()
                total_batches += 1
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        cer_percent = (cer_sum / total_chars) * 100 if total_chars > 0 else 0.0
        return avg_loss, cer_percent, best_cer

    def dev_epoch(self):
        total_loss = 0.0
        total_batches = 0
        cer_sum = 0
        total_chars = 0
        self.model.eval()
        with sdpa_kernel(SDPBackend.MATH):
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dev_loader, ncols=80, desc="Development")):
                    features = batch['features'].to(self.device)
                    feature_lengths = batch['feature_lengths'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    target_lengths = batch['target_lengths'].to(self.device)

                    with self.amp:
                        logits, out_lens = self.model(features, feature_lengths)
                        log_probs = functional.log_softmax(logits, dim=-1).transpose(0, 1)  # (time, batch, vocab_size)
                    loss = self.loss(log_probs, targets, out_lens, target_lengths)
                    loss = loss / features.size(0)
                    pred_ids_batch = ctc_greedy_decode_ids(
                        logit_batch=logits,
                        lengths=out_lens,
                        blank_id=0
                    )
                    offset = 0
                    for b in range(len(pred_ids_batch)):
                        cur_len = int(target_lengths[b].item())
                        target_ids = targets[offset:offset+cur_len].tolist()
                        offset += cur_len
                        target_str = self.char_tokenizer.decode(target_ids)
                        pred_str = self.char_tokenizer.decode(pred_ids_batch[b])
                        cer = char_edit_distance(target_str, pred_str)
                        cer_sum += cer
                        total_chars += len(target_str)
                    total_loss += loss.item()
                    total_batches += 1
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        cer_percent = (cer_sum / total_chars) * 100 if total_chars > 0 else 0.0
        return avg_loss, cer_percent

    def train_main(self):
        for epoch in range(1, self.yaml_config['epoch'] + 1):
            train_loss, train_cer, best_cer = self.train_epoch()
            dev_loss, dev_cer = self.dev_epoch()
            print(
                f"Epoch {epoch}/{self.yaml_config['epoch']}: "
                f"Train Loss={train_loss:.4f}, CER={train_cer:.2f}% | "
                f"Dev Loss={dev_loss:.4f}, CER={dev_cer:.2f}% | "
                f"Best Train CER={best_cer:.2f}%"
            )
            self.train_sampler.set_epoch(epoch)
            self.dev_sampler.set_epoch(epoch)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_main()
