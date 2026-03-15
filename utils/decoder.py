import torch
from typing import List

def ctc_greedy_decode_ids(
            logit_batch:torch.Tensor,
            lengths:torch.Tensor,
            blank_id:int=0
        ):
        with torch.no_grad():
            pred_ids = torch.argmax(logit_batch, dim=-1)
        results:List[List[int]] = []
        for b in range(logit_batch.size(0)):
            pred_id_seq = pred_ids[b][:lengths[b]].tolist()
            prev_id = None
            decoded_ids = []
            for pid in pred_id_seq:
                if pid != blank_id and pid != prev_id:
                    decoded_ids.append(pid)
                prev_id = pid
            results.append(decoded_ids)
        return results