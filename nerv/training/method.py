import os
from torch._C import device
import wandb
import numpy as np

import torch
import torch.nn as nn

from nerv.utils.io import check_file_exist
from nerv.training.lr import get_lr


class BaseMethod(nn.Module):
    """Base method for training a model.

    Args:
        model (`BaseModel`): the network to train. See `model.py`.
        datamodule (`BaseDataModule`): the datamodule with train/val loader.
            See `datamodule.py`.
        params (`BaseParams`): the parameter settings. See `params.py`.
        ckp_path (str): root to save the checkpoints.
        local_rank (int, optional): used in DDP training. Default: 0.
        use_ddp (bool, optional): whether use DDP training. Default: False.
        use_fp16 (bool, optional): whether use FP16 mixed precision training.
            Default: False.

    Member Variables:
        # model
        `self.model`: as `model`.

        # optimization
        `self.optimizer`: optimizer.
        `self.scheduler`: lr scheduler.
        `self.scheduler_method`: determine how to adjust lr, 'step' or 'epoch'.
        `self.clip_grad`: clipping gradient value. Not clipping if <= 0.

        # data related
        `self.train_loader`: `datamodule.train_loader`.
        `self.val_loader`: `datamodule.val_loader`.

        # statistics
        `self.it`: total training iterations.
        `self.epoch`: total training epochs.
        `self.print_iter`: interval between print/log training statistics.
        `self.save_iter`: interval between saving checkpoint.
        `self.epoch_it`: iteration number in one epoch.
        `self.stats_dict`: accumulate values. Can be used for avg.

        # all parameter settings
        `self.params`: as `params`.
        `self.ckp_path`: as `ckp_path`.
        `self.local_rank`: as `local_rank`.

    """

    def __init__(self,
                 model,
                 datamodule,
                 params,
                 ckp_path,
                 local_rank=0,
                 use_ddp=False,
                 use_fp16=False):
        super().__init__()
        self.model = model
        self.params = params

        # DDP training
        self.use_ddp = use_ddp
        self.local_rank = local_rank
        self._init_env()

        # FP16 mixed precision training
        self.use_fp16 = use_fp16

        # data
        self.train_loader = datamodule.train_loader
        self.val_loader = datamodule.val_loader

        # training settings
        self.ckp_path = ckp_path
        self._init_training()

    def _init_env(self):
        """Init training environment related settings.

        - `self.device`
        - DDP training
        """
        self.gpus = torch.cuda.device_count()
        assert self.gpus == self.params.gpus, 'GPU number not aligned'

        if not self.use_ddp:
            assert self.gpus == 1, 'Multi-GPU training should use DDP'
            assert self.local_rank == 0, \
                'local rank should be 0 in non DDP training'

        # DDP init
        torch.cuda.set_device(self.local_rank)
        torch.distributed.init_process_group('nccl', init_method='env://')
        self.device = torch.device(f'cuda:{self.local_rank}')

        # model to device
        self.model = self.model.to(device)
        if self.use_ddp:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank])
        else:
            # this is only to enable calling `self.model.module`
            self.model = nn.parallel.DataParallel(
                self.model, device_ids=self.gpus)

    def _init_amp(self):
        """Init FP16 mixed precision training related settings."""
        if not self.use_fp16:
            return

        self.grad_scaler = torch.cuda.amp.GradScaler()

    def _init_training(self):
        # training accumulator
        self.it, self.epoch = 0, 0
        self.max_epochs = self.params.max_epochs
        self.print_iter = self.params.print_iter
        self.save_iter = int(
            np.ceil(len(self.train_loader) * self.params.save_interval))

        # gradient clipping
        self.clip_grad = self.params.clip_grad

        # construct optimizer and lr_scheduler
        self._setup_optimizer()

        # automatically detect existing checkpoints
        self.load_ckp()

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def fit(self, san_check_val_step=2):
        """Train the model.

        Args:
            san_check_val_step (int): run a few val steps to verify the model,
                logging, checkpointing etc implementations.
        """
        # run several val steps as sanity check
        if san_check_val_step > 0:
            self.validation_epoch(san_check_step=san_check_val_step)

        for _ in range(self.epoch, self.max_epochs):
            self.training_epoch()

    def training_epoch(self):
        """Train one epoch.

        - Calculate loss via `self.training_step`
        - Backward and perform one optimization step
        - Apply lr scheduler
        - Accumulate iter and epoch number
        """
        self._training_epoch_start()

        for batch_idx, batch_data in enumerate(self.train_loader):
            # set the batch idx
            self.epoch_it = batch_idx

            self._training_step_start()

            # run model forward and calculate losses
            if self.use_fp16:
                with torch.autocast():
                    out_dict = self._training_step(batch_data)
            else:
                out_dict = self._training_step(batch_data)

            # backward and optimize step
            if self.use_fp16:
                self._optimize_train_fp16(out_dict['loss'])
            else:
                self._optimize_train(out_dict['loss'])

            # logging
            self._log_train(out_dict)

            self._training_step_end()

        self._training_epoch_end()

    def _training_step(self, batch):
        """Returns a dict containing 'loss' to apply optimizer on.
        The values in the dict will be logged to wandb.
        """
        train_loss = self.model.module.loss_function(batch)
        loss = torch.tensor(0.).to(self.device)
        for loss_name, loss_value in train_loss.items():
            assert loss_name.endswith('_loss')
            loss = loss + loss_value * eval(f'self.params.{loss_name}_w')
        train_loss['loss'] = loss
        return train_loss

    def _optimize_train(self, loss):
        """Clip the gradient of model."""
        loss.backward()
        if self.clip_grad > 0.:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _optimize_train_fp16(self, loss):
        """Clip the gradient of model."""
        self.grad_scaler.scale(loss).backward()
        if self.clip_grad > 0.:
            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        # only log results for rank 0
        if self.local_rank != 0:
            return

        # accumulate results till print interval
        if (self.epoch_it + 1) % self.print_iter != 0:
            self._accumulate_stats(out_dict)
            return

        out_dict = {
            f'train/{k}': np.mean(v)
            for k, v in self.stats_dict.items()
        }
        out_dict['train/epoch'] = self.epoch
        out_dict['train/it'] = self.it
        out_dict['train/lr'] = get_lr(self.optimizer)
        wandb.log(out_dict, step=self.it)
        self.stats_dict = None

    def _training_epoch_start(self):
        """Things to do at the beginning of every training epoch."""
        self.optimizer.zero_grad()
        self.model.train()
        self.stats_dict = None

    def _training_step_start(self):
        """Things to do at the beginning of every training step."""
        pass

    def _training_step_end(self):
        """Things to do at the end of every training step."""
        if self.scheduler_method == 'step':
            self.scheduler.step()
        self.it += 1
        if (self.epoch_it + 1) % self.save_iter == 0:
            self.save_ckp()
        if self.use_ddp:
            torch.distributed.barrier()

    def _training_epoch_end(self):
        """Things to do at the end of every training epoch."""
        if self.scheduler_method == 'epoch':
            self.scheduler.step()
        self.epoch += 1
        self.stats_dict = None

        # run one epoch of validation after each training epoch
        self.validation_epoch()

    @torch.no_grad()
    def validation_epoch(self, san_check_step=-1):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        # only do evaluation for rank 0
        if self.local_rank != 0:
            return

        self._validation_epoch_start()

        for batch_idx, batch_data in enumerate(self.val_loader):
            out_dict = self._validation_step(batch_data)

            self._accumulate_stats(out_dict)

            # if it's actually in sanity check
            if san_check_step > 0 and batch_idx + 1 >= san_check_step:
                break

        out_dict = {f'val/{k}': np.mean(v) for k, v in self.stats_dict.items()}
        wandb.log(out_dict, step=self.it)

        self._validation_epoch_end()

    def _validation_step(self, batch):
        """Returns a dict containing losses to log."""
        val_loss = self.model.module.loss_function(batch)
        return val_loss

    def _validation_epoch_start(self):
        """Things to do at the beginning of every validation epoch."""
        self.model.eval()
        self.stats_dict = None

    def _validation_epoch_end(self):
        """Things to do at the end of every validation epoch."""
        self.save_ckp()
        self.stats_dict = None

    def _setup_optimizer(self):
        self.optimizer, (self.scheduler, self.scheduler_method) = \
            self._configure_optimizers()
        assert self.scheduler_method in ['step', 'epoch', '']
        if not self.scheduler_method:
            assert self.optimizer is None

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        pass

    def _accumulate_stats(self, stats_dict):
        """Append stats in `stats_dict` to `self.stats_dict`.

        We assume that each value in stats_dict is a torch scalar.
        """
        if self.stats_dict is None:
            self.stats_dict = {k: [v.item()] for k, v in stats_dict.items()}
        else:
            for k, v in stats_dict.items():
                self.stats_dict[k].append(v.item())

    def save_ckp(self, ckp_path=None):
        """Save state_dict of all self.modules.

        The default save name is '{self.ckp_path}/model_{self.it}.pth'.
        """
        # only save model for rank 0
        if self.local_rank != 0:
            return

        if ckp_path is None:
            ckp_path = os.path.join(self.ckp_path, f'model_{self.it}.pth')
        ckp = {
            'state_dict': self.model.module.state_dict(),
            'opt_state_dict': self.optimizer.state_dict(),
            'it': self.it,
            'epoch': self.epoch,
        }
        if self.scheduler_method:
            ckp['scheduler_state_dict'] = self.scheduler.state_dict()
            ckp['scheduler_method'] = self.scheduler_method
        if self.use_fp16:
            ckp['grad_scaler'] = self.grad_scaler.state_dict()

    def load_ckp(self, ckp_path=None, auto_detect=True):
        """Load from checkpoint.

        Support automatic detection of existing checkpoints.
        Useful in SLURM preemption systems.
        """
        # automatically detect checkpoints
        if auto_detect and os.path.exists(self.ckp_path):
            ckp_files = os.listdir(self.ckp_path)
            ckp_files = [ckp for ckp in ckp_files if ckp.endswith('.pth')]
            if ckp_files:
                ckp_files = sorted(
                    ckp_files,
                    key=lambda x: os.path.getmtime(
                        os.path.join(self.ckp_path, x)))
                last_ckp = ckp_files[-1]
                print(f'INFO: automatically detect checkpoint {last_ckp}')
                ckp_path = os.path.join(self.ckp_path, last_ckp)

        if not ckp_path:
            return

        check_file_exist(ckp_path)
        ckp = torch.load(ckp_path)
        self.it, self.epoch = ckp['it'], ckp['epoch']
        self.model.module.load_state_dict(ckp['state_dict'])
        self.optimizer.load_state_dict(ckp['opt_state_dict'])
        if self.scheduler_method:
            self.scheduler.load_state_dict(ckp['scheduler_state_dict'])
            self.scheduler_method = ckp['scheduler_method']
        if self.use_fp16:
            self.grad_scaler.load_state_dict(ckp['grad_scaler'])
