"""Template train.py"""

import os
import importlib
import argparse
import wandb

from datamodule import BaseDataModule
from method import BaseMethod
from model import BaseModel

from nerv.utils.io import mkdir_or_exist


def main(params):
    if args.fp16:
        print("INFO: using FP16 training!")
    if args.ddp:
        print("INFO: using DDP training!")

    # train_set, val_set = build_dataset()
    datamodule = BaseDataModule(
        params, train_set=None, val_set=None, use_ddp=args.ddp)

    model = BaseModel()

    if args.local_rank == 0:
        exp_name = f'{args.params}-fp16' if args.fp16 else args.params
        ckp_path = os.path.join(CHECKPOINT, exp_name, 'models')
        mkdir_or_exist(os.path.dirname(ckp_path))

        # on clusters, quota is limited
        # soft link temp space for checkpointing
        if not os.path.exists(ckp_path):
            os.system(r'ln -s /checkpoint/ziyiwu/{}/ {}'.format(
                SLURM_JOB_ID, ckp_path))

        wandb.init(project=params.project, name=exp_name, id=exp_name)

    method = BaseMethod(
        model,
        datamodule,
        params,
        ckp_path=ckp_path,
        local_rank=args.local_rank,
        use_ddp=args.ddp,
        use_fp16=args.fp16,
    )

    method.fit(
        resume_from=args.weight, san_check_val_step=params.san_check_val_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NERV model training')
    parser.add_argument('--params', type=str, default='params')
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    params = importlib.import_module(args.params)
    params = params.BaseParams()
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    CHECKPOINT = './checkpoint/'
    main(params)
