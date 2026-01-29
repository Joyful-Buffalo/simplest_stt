from typing import Any, Dict, List, Tuple
import torch


def asr_collate_fn(batch:List[Tuple[torch.Tensor, torch.Tensor]])->Dict[str, Any]:
    fbank, idxs = zip(*batch)
    feat_lengths = torch.tensor([f.shape[0] for f in fbank], dtype=torch.long)
    padded_fbank = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True)
    target_lengths = torch.tensor([len(i) for i in idxs], dtype=torch.long)
    concatenated_idxs = torch.cat(idxs, dim=0)
    return {
        "features": padded_fbank,
        "feature_lengths": feat_lengths,
        "targets": concatenated_idxs,
        "target_lengths": target_lengths
    }