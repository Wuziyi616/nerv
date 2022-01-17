class BaseParams:
    project = 'NERV'  # project name for wandb logging

    # training settings
    gpus = 1
    ddp_unused_params = False
    max_epochs = 100
    san_check_val_step = 2
    print_iter = 50
    save_interval = 1.0
    eval_interval = 1

    # optimizer settings
    lr = 1e-3
    weight_decay = 0.0
    clip_grad = -1

    # data settings
    data_root = ''
    train_batch_size = 64
    val_batch_size = 64
    num_workers = 4

    def to_dict(self):
        all_vars = [var for var in dir(self) if not var.startswith('__')]
        all_vars.remove('to_dict')
        all_vars.remove('from_dict')
        return {var: getattr(self, var) for var in all_vars}

    @staticmethod
    def from_dict(params_dict):
        params = BaseParams()
        for k, v in params_dict.items():
            setattr(params, k, v)
        return params
