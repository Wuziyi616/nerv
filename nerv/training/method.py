import os
import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from nerv.utils.io import check_file_exist
from nerv.utils.misc import AverageMeter
from nerv.utils.tensor import ddp_all_gather
from nerv.utils.conversion import is_list_of
from nerv.training.lr import get_lr
from nerv.models.utils import filter_wd_parameters


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
        `self.grad_scaler`: used in FP16 mixed precision training.
        `self.scheduler`: lr scheduler.
        `self.scheduler_method`: determine how to adjust lr, 'step' or 'epoch'.
        `self.clip_grad`: clipping gradient value. Not clipping if <= 0.

        # data related
        `self.train_loader`: `datamodule.train_loader`.
        `self.iter_train_loader`: `iter(self.train_loader)`, used for saving
            data_loader's state in checkpoint.
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
        `self.use_ddp`: as `use_ddp`.
        `self.use_fp16`: as `use_fp16`.
        `self.gpus`: number of GPUs available, should equal to `params.gpus`.
        `self.device`: device of current process, e.g. `cuda:0`.

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
        self._init_amp()

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

        # DDP init
        if self.use_ddp:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group('nccl', init_method='env://')
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cuda')
            assert self.gpus == 1, 'Multi-GPU training should use DDP'
            assert self.local_rank == 0, \
                'local rank should be 0 in non DDP training'

        # model to device
        self.model = self.model.to(self.device)
        if self.use_ddp:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.params.ddp_unused_params)
        else:
            # this is only to enable calling `self.model.module`
            self.model = nn.parallel.DataParallel(
                self.model, device_ids=list(range(self.gpus)))

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
            np.ceil(len(self.train_loader) * self.params.save_interval)) + 1
        self.eval_interval = self.params.eval_interval

        # gradient clipping
        self.clip_grad = self.params.clip_grad

        # construct optimizer and lr_scheduler
        self._setup_optimizer()

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def fit(self, resume_from='', san_check_val_step=2):
        """Train the model.

        Args:
            resume_from (str, optional): pre-trained weight path.
                Default: ''.
            san_check_val_step (int, optional): run a few val steps to verify
                the model, logging, checkpointing etc implementations.
                Default: 2.
        """
        # automatically detect existing checkpoints
        self.load_ckp(ckp_path=resume_from)

        # run several val steps as sanity check
        if self.local_rank == 0 and san_check_val_step > 0:
            self.validation_epoch(
                self.model.module, san_check_step=san_check_val_step)

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

        # use iter dataloader in order to save its state
        self.iter_train_loader = iter(self.train_loader)
        train_steps = (
            len(self.train_loader.sampler) -
            self.train_loader.sampler.real_counter(
                self.iter_train_loader)) // self.params.train_batch_size
        with tqdm(total=train_steps, desc=f'Train epoch {self.epoch}') as t:
            for batch_idx, batch_data in enumerate(self.iter_train_loader):
                torch.cuda.empty_cache()
                # set the batch idx
                self.epoch_it = batch_idx
                batch_data = {
                    k: v.to(self.device)
                    for k, v in batch_data.items()
                }

                self._training_step_start()

                # model forward, loss computation, backward and optimize
                out_dict = self._training_step(batch_data)

                # logging
                self._log_train(out_dict)

                self._training_step_end()

                t.set_postfix(loss=f"{out_dict['loss'].item():.4f}")
                t.update(1)

        self._training_epoch_end()

    def _loss_function(self, batch_data):
        """Compute and aggregate losses."""
        out_dict = self.model.module.loss_function(batch_data)
        assert 'loss' not in out_dict.keys()
        loss = torch.tensor(0.).to(self.device)
        for loss_name, loss_value in out_dict.items():
            if not loss_name.endswith('_loss'):
                continue
            loss = loss + loss_value * eval(f'self.params.{loss_name}_w')
        out_dict['loss'] = loss
        return out_dict

    def _training_step_fp32(self, batch_data):
        """Loss backward and optimize in the normal FP32 setting."""
        out_dict = self._loss_function(batch_data)
        loss = out_dict['loss']
        loss.backward()
        if self.clip_grad > 0.:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return out_dict

    def _training_step_fp16(self, batch_data):
        """Loss backward and optimize in FP16 mixed precision setting."""
        with torch.cuda.amp.autocast():
            out_dict = self._loss_function(batch_data)
        loss = out_dict['loss']
        self.grad_scaler.scale(loss).backward()
        if self.clip_grad > 0.:
            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()
        return out_dict

    def _training_step(self, batch_data):
        """Returns a dict containing 'loss' to apply optimizer on.
        The values in the dict will be logged to wandb.
        """
        if self.use_fp16:
            return self._training_step_fp16(batch_data)
        return self._training_step_fp32(batch_data)

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        # only log results for rank 0
        if self.local_rank != 0:
            return

        # accumulate results till print interval
        self._accumulate_stats(out_dict)

        if (self.epoch_it + 1) % self.print_iter != 0:
            return

        out_dict = {f'train/{k}': v.avg for k, v in self.stats_dict.items()}
        out_dict['train/epoch'] = self.epoch
        out_dict['train/it'] = self.it
        out_dict['train/lr'] = get_lr(self.optimizer)
        if self.use_fp16:
            out_dict['train/fp16_loss_scale'] = self.grad_scaler.get_scale()
        wandb.log(out_dict, step=self.it)
        self.stats_dict = None

    def _training_epoch_start(self):
        """Things to do at the beginning of every training epoch."""
        self.optimizer.zero_grad()
        self.train()
        self.stats_dict = None
        print(f'>>> Training epoch {self.epoch} start')

        # sync DDP training processes at the end of epoch
        if self.use_ddp:
            torch.distributed.barrier()

    def _training_step_start(self):
        """Things to do at the beginning of every training step."""
        pass

    def _training_step_end(self):
        """Things to do at the end of every training step."""
        if self.scheduler_method == 'step':
            self.scheduler.step()
        self.it += 1
        if (self.epoch_it + 1) % self.save_iter == 0:
            self.save_ckp(save_loader=True)

        # sync DDP training processes at the end of epoch
        if self.use_ddp:
            torch.distributed.barrier()

    def _training_epoch_end(self):
        """Things to do at the end of every training epoch."""
        if self.scheduler_method == 'epoch':
            self.scheduler.step()
        self.epoch += 1
        self.stats_dict = None
        self.save_ckp(save_loader=False)

        # run one epoch of validation after each training epoch
        if self.local_rank == 0 and (self.epoch + 1) % self.eval_interval == 0:
            self.validation_epoch(self.model.module)

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.

        According to a discussion, in DDP eval we should pass the inner module
            of DDPModel for eval to prevent hanging.
        See: https://discuss.pytorch.org/t/torch-distributed-barrier-hangs-in-ddp/114522.  # noqa
        """
        print('>>> Evaluating')
        model.eval()
        self.stats_dict = None
        torch.cuda.empty_cache()

        for batch_idx, batch_data in enumerate(
                tqdm(self.val_loader, desc=f'Eval epoch {self.epoch}')):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}

            out_dict = model.loss_function(batch_data)
            self._accumulate_stats(out_dict)

            # if it's actually in sanity check
            if san_check_step > 0 and batch_idx + 1 >= san_check_step:
                self.stats_dict = None
                torch.cuda.empty_cache()
                return

        # log eval statistics
        out_dict = {f'val/{k}': v.avg for k, v in self.stats_dict.items()}
        wandb.log(out_dict, step=self.it)
        self.stats_dict = None
        torch.cuda.empty_cache()

    def _setup_optimizer(self):
        """Construct optimizer and lr scheduler."""
        self.optimizer, (self.scheduler, self.scheduler_method) = \
            self._configure_optimizers()
        assert self.scheduler_method in ['step', 'epoch', '']
        if not self.scheduler_method:
            assert self.optimizer is None

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        lr = self.params.lr
        wd = self.params.weight_decay
        if wd > 0.:
            params_dict = filter_wd_parameters(self.model)
            params_list = [{
                'params': params_dict['no_decay'],
                'weight_decay': 0.,
            }, {
                'params': params_dict['decay'],
                'weight_decay': wd,
            }]
            # use AdamW in weight_decay
            optimizer = optim.AdamW(params_list, lr=lr)
        else:
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                weight_decay=0.)
        return optimizer, (None, '')

    @torch.no_grad()
    def _accumulate_stats(self, stats_dict):
        """Append stats in `stats_dict` to `self.stats_dict`.

        We assume that each value in stats_dict is a torch scalar.
        """
        bs = stats_dict.pop('batch_size', 1)
        if self.stats_dict is None:
            self.stats_dict = {k: AverageMeter() for k in stats_dict.keys()}
        for k, v in stats_dict.items():
            self.stats_dict[k].update(v.item(), bs)

    @torch.no_grad()
    def _gather_train_sampler_state(self):
        """Gather the state for self.train_loader across DDP."""
        sampler_state = self.train_loader.sampler.state_dict(
            self.iter_train_loader)
        indices = sampler_state['indices']  # a list of int
        counter = sampler_state['counter']  # an int
        assert is_list_of(indices, int) and isinstance(counter, int)

        # not in DDP, directly get one and return
        if not self.use_ddp:
            return {
                'rank0_train_sampler': {
                    'indices': indices,
                    'counter': counter
                },
            }

        # need to gather all rank dataloader in DDP
        # all_gather only supports tensor, so we convert them into tensor
        sampler_state = indices + [counter]
        sampler_state = torch.tensor(sampler_state).to(self.device)
        gather_state = ddp_all_gather(sampler_state).cpu()
        indices, counter = gather_state[:, :-1], gather_state[:, -1]
        # lists of shape [world_size, N], [world_size]
        indices, counter = indices.tolist(), counter.tolist()
        assert len(indices) == len(counter) == dist.get_world_size()
        return {
            f'rank{i}_train_sampler': {
                'indices': indices[i],
                'counter': counter[i]
            }
            for i in range(len(counter))
        }

    @torch.no_grad()
    def save_ckp(self, save_loader=False, keep_num=5):
        """Save state_dict of all self.modules.

        The default save name is '{self.ckp_path}/model_{self.it}.pth'.
        """
        # if at the middle of training, should save dataloader states
        # if in eval, no need to do so
        if save_loader:
            train_sampler_states = self._gather_train_sampler_state()

        # only rank 0 process save ckp
        if self.local_rank != 0:
            return

        # auto remove earlier ckps
        ckp_files = os.listdir(self.ckp_path)
        ckp_files = [ckp for ckp in ckp_files if ckp.endswith('.pth')]
        if keep_num > 0 and len(ckp_files) >= keep_num:
            ckp_files = sorted(
                ckp_files,
                key=lambda x: os.path.getmtime(os.path.join(self.ckp_path, x)))
            del_ckp = ckp_files[:-(keep_num - 1)]
            for x in del_ckp:
                os.remove(os.path.join(self.ckp_path, x))

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
        if save_loader:
            ckp.update(train_sampler_states)
        torch.save(ckp, ckp_path)

    @torch.no_grad()
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

        print(f'INFO: loading checkpoint {ckp_path}')
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
        # should consider loading data sampler
        if 'rank0_train_sampler' in ckp.keys():
            print('INFO: loading train loader state')
            self.train_loader.sampler.load_state_dict(
                ckp[f'rank{self.local_rank}_train_sampler'])
