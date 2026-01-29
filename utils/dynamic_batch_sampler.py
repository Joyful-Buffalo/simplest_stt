import torch
from torch.utils.data import Sampler
from typing import Iterator, List, Optional, Sequence


class DynamicBatchSampler(Sampler[List[int]]):
    def __init__(
            self, 
            lengths: Sequence[int], 
            batch_size: Optional[int] = None,
            max_frame_per_batch: Optional[int] = None,
            shuffle: bool = True,
            bucket_size: int = 100,
            generator: Optional[torch.Generator] = None,
            seed: Optional[int] = None,
            drop_last: bool = False,
            allow_oversize_single: bool = True
            ):
        assert (batch_size is not None) ^ (max_frame_per_batch is not None), "Either batch_size or max_frame_per_batch must be provided, but not both."
        assert bucket_size > 0, "bucket_size must be positive."
        self.lengths = lengths
        self.batch_size = batch_size
        self.max_frame_per_batch = max_frame_per_batch
        self.shuffle = shuffle
        self.bucket_size = bucket_size
        self.allow_oversize_single = allow_oversize_single
        self.indices = list(range(len(lengths)))
        self.drop_last = drop_last

        if seed is not None:
            self._base_seed = seed
        elif generator is not None:
            self._base_seed = generator.initial_seed()
        else:
            self._base_seed = torch.initial_seed()

        self._epoch = 0
        self._cached_epoch = None
        self._cached_indices: List[int] | None= None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._cached_epoch = None
        self._cached_indices = None

    def _make_epoch_generator(self) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self._base_seed + self._epoch)
        return g

    def _order_indices(self)-> List[int]:
        if self._cached_epoch == self._epoch and self._cached_indices is not None:
            return self._cached_indices

        if self.shuffle:
            indices = torch.randperm(len(self.lengths), generator=self._make_epoch_generator()).tolist()
        else:
            indices = self.indices.copy()
        buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        for bucket in buckets:
            bucket.sort(key=self.lengths.__getitem__)
        ordered_indices = [idx for bucket in buckets for idx in bucket]
        self._cached_epoch = self._epoch
        self._cached_indices = ordered_indices
        return ordered_indices

    def __iter__(self)-> Iterator[List[int]]:
        indices = self._order_indices()
        if self.batch_size is not None:
            n_full_frames = (len(indices) // self.batch_size) * self.batch_size
            for i in range(0, n_full_frames, self.batch_size):
                yield indices[i:i + self.batch_size]
            if not self.drop_last and n_full_frames < len(indices):
                yield indices[n_full_frames:]
            return

        batch = []
        total_frames = 0
        for idx in indices:
            length = self.lengths[idx]
            if length > self.max_frame_per_batch:
                if batch:
                    yield batch
                    batch = []
                    total_frames = 0
                if self.allow_oversize_single:
                    yield [idx]
                else:
                    raise ValueError(f"Sample length {length} exceeds max_frame_per_batch {self.max_frame_per_batch} and allow_oversize_single is False.")
                continue
            if total_frames + length <= self.max_frame_per_batch:
                batch.append(idx)
                total_frames += length
            else:
                if batch:
                    yield batch
                batch = [idx]
                total_frames = length
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self)-> int:
        if self.batch_size is not None:
            if self.drop_last:
                return len(self.lengths) // self.batch_size
            return (len(self.lengths) + self.batch_size - 1) // self.batch_size
        total = 0
        cnt = 0
        has_batch = False
        for idx in self._order_indices():
            L = self.lengths[idx]
            if L > self.max_frame_per_batch:
                if has_batch:
                    cnt += 1
                    has_batch = False
                    total = 0
                if self.allow_oversize_single:
                    cnt += 1
                else:
                    raise ValueError(f"Sample length {L} exceeds max_frame_per_batch {self.max_frame_per_batch} and allow_oversize_single is False.")
                continue
            if total + L <= self.max_frame_per_batch:
                total += L
                has_batch = True
            else:
                if has_batch:
                    cnt += 1
                total = L
                has_batch = True
        if has_batch and not self.drop_last:
            cnt += 1
        return cnt