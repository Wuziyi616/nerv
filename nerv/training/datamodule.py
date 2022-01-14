import torch
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler


class StatefulSampler(sampler.Sampler):
    """Stateful sampler that supports checkpointing."""

    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle

        # initialize dataloader index
        self._init_index()

    def _init_index(self):
        # counter
        self.counter = 0

        # construct index
        if self.shuffle:
            self.indices = torch.randperm(len(self.dataset))
        else:
            self.indices = torch.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.indices)

    def __next__(self):
        if self.counter == len(self.indices):
            self._init_index()
            raise StopIteration()
        else:
            idx = self.indices[self.counter]
            self.counter += 1
            return int(idx)

    def state_dict(self, iter_loader=None):
        """iter_loader: iter(DataLoader) instance."""
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if iter_loader is not None:
            if iter_loader._num_workers > 0:
                bs = iter_loader._index_sampler.batch_size
                prefetched_num = \
                    (iter_loader._send_idx - iter_loader._rcvd_idx) * bs
        return {
            'indices': self.indices,
            'counter': self.counter - prefetched_num,
        }

    def load_state_dict(self, state_dict):
        self.indices = state_dict['indices']
        self.counter = state_dict['counter']


class StatefulDistributedSampler(DistributedSampler):
    """Distributed version of StatefulSampler."""

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 drop_last=True):
        assert drop_last, 'Currently only supporting `drop_last=True`'
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last)

        # initialize dataloader index
        self._init_index()

    def _init_index(self):
        # counter
        self.counter = 0

        # construct index
        if self.shuffle:
            indices = torch.randperm(len(self.dataset))
        else:
            indices = torch.arange(len(self.dataset))
        indices = indices.tolist()

        # since we do `drop_last`
        # remove tail of data to make it evenly divisible.
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        self.indices = indices

    def __iter__(self):
        """Returns an iterator `it` that `next(it)` gives the next data idx."""
        return self

    def __next__(self):
        if self.counter == len(self.indices):
            self._init_index()
            raise StopIteration()
        else:
            idx = self.indices[self.counter]
            self.counter += 1
            return int(idx)

    def state_dict(self, iter_loader=None):
        """iter_loader: iter(DataLoader) instance."""
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if iter_loader is not None:
            if iter_loader._num_workers > 0:
                bs = iter_loader._index_sampler.batch_size
                prefetched_num = \
                    (iter_loader._send_idx - iter_loader._rcvd_idx) * bs
        return {
            'indices': self.indices,
            'counter': self.counter - prefetched_num,
        }

    def load_state_dict(self, state_dict):
        self.indices = state_dict['indices']
        self.counter = state_dict['counter']


class BaseDataModule:

    def __init__(self, params, train_set, val_set, use_ddp=False):
        self.params = params
        self.train_set = train_set
        self.val_set = val_set
        self.use_ddp = use_ddp

        self._build_dataloader()

    def _build_dataloader(self):
        """Build training and validation data loaders."""
        if self.use_ddp:
            state_dist_sampler = StatefulDistributedSampler(
                self.train_set, shuffle=True, drop_last=True)
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=self.params.train_batch_size,
                sampler=state_dist_sampler,
                num_workers=self.params.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )
        else:
            state_sampler = StatefulSampler(self.train_set, shuffle=True)
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=self.params.train_batch_size,
                sampler=state_sampler,
                num_workers=self.params.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )

        val_sampler = StatefulSampler(self.val_set, shuffle=False)
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.params.val_batch_size,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
