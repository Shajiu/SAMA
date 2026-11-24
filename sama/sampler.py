"""Grouped batch sampler for SAMA.

Every batch is homogeneous in (adapter_group, is_low_resource, is_unsafe) so that:

* ``adapter_group``          -> exactly one Script-LoRA branch is active per batch,
                                which keeps the inactive branches free of gradients
                                (required because DDP runs with
                                ``find_unused_parameters=True``);
* ``is_low_resource``        -> the structural / robustness views can be computed for
                                the whole batch or skipped for the whole batch;
* ``is_unsafe``              -> safety batches are contiguous, so that the extra
                                refusal forward can actually be scheduled on all ranks
                                in the same step (see ``SAMASeq2SeqTrainer``).

Deliberately dependency-free (no torch.distributed), because the sampler is also
used by the data-building / validation tools.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from torch.utils.data import Sampler


class GroupedBatchSampler(Sampler[list[int]]):
    """Build homogeneous batches so each batch activates exactly one adapter branch."""

    def __init__(self, dataset, batch_size: int, seed: int = 42, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.groups = defaultdict(list)
        for idx in range(len(dataset)):
            row = dataset[idx]
            key = (
                str(row["adapter_group"]),
                bool(row["is_low_resource"]),
                bool(row.get("is_unsafe", False)),
            )
            self.groups[key].append(idx)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches = []
        for indices in self.groups.values():
            indices = list(indices)
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        total = 0
        for indices in self.groups.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total
