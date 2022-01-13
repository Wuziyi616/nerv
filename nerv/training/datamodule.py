from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class BaseDataModule:

    def __init__(self, params, use_ddp=False):
        self.params = params
        self.use_ddp = use_ddp

        # self.train_set = None
        # self.val_set = None

        self._build_dataloader()

    def _build_dataloader(self):
        """Build training and validation data loaders."""
        if self.use_ddp:
            dist_sampler = DistributedSampler(
                self.train_set, shuffle=True, drop_last=True)
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=self.params.train_batch_size,
                num_workers=self.params.num_workers,
                sampler=dist_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
        else:
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=self.params.train_batch_size,
                shuffle=True,
                num_workers=self.params.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )

        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.params.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
