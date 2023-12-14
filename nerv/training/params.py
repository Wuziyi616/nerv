class BaseParams(object):
    project = 'NERV'  # project name for wandb logging

    # training settings
    gpus = 1
    ddp_unused_params = False
    max_epochs = 100
    san_check_val_step = 2  # to verify code correctness
    print_iter = 50
    save_interval = 1.0  # save every (num_epoch_iters * save_interval) iters
    eval_interval = 1  # should be int, number of epochs between each eval
    save_epoch_end = False  # save ckp at the end of every epoch
    n_samples = 5  # number of visualizations after each evaluation

    # optimizer settings
    optimizer = 'Adam'
    lr = 1e-3
    weight_decay = 0.0
    clip_grad = -1

    # data settings
    data_root = ''
    train_batch_size = 64 // gpus  # batch size per gpu
    val_batch_size = train_batch_size * 2
    num_workers = 8  # workers per gpu
    grad_accum_steps = 1  # gradient accumulation

    # loss configs
    # we need to have `xxx_loss` as a key in the returned dict from the
    # `calc_train_loss` function in model
    # xxx_loss_w = 1.

    # we support metric monitoring when saving checkpoints
    ckp_monitor = ''  # e.g. 'val/miou'
    ckp_monitor_type = 'max'  # 'max' or 'min'
    copy_ckp_end = False  # the ckp_path might be purged after training ends
    # we support copying the last/best ckp to a new path
    # if `True`, move it to `ckp_path/../`
    # or you can specify a new path by setting it to a str

    # we support changing some variables values with epoch
    # begin with 1., change to 10. at 40th epoch and then 100. at 80th epoch
    # var1 = 1.
    # var1_all = [1., 10., 100.]
    # var1_t = [40, 80]

    def __getitem__(self, key):
        """`__getitem__` function similar to dict."""
        return getattr(self, key)

    def __str__(self):
        """`__str__` function similar to dict."""
        all_vars = self._get_all_var_names()
        # format it in a nice way
        max_len = max([len(var) for var in all_vars])
        return f'class {self.project}Params:\n' + '\n'.join(
            [f'{var:>{max_len}}: {getattr(self, var)}' for var in all_vars])

    def _get_all_var_names(self):
        """Get all variable names."""
        all_vars = [var for var in dir(self) if not var.startswith('__')]
        # remove class methods (callable functions)
        # weird, cannot filter out `get` by checking callable
        #   thus manually removing it
        all_vars.remove('get')
        for var in all_vars:
            if callable(getattr(self, var)):
                all_vars.remove(var)
        return all_vars

    def get(self, key, value=None):
        """`get` function similar to dict."""
        if hasattr(self, key):
            return getattr(self, key)
        return value

    def to_dict(self):
        """Convert to dict."""
        all_vars = self._get_all_var_names()
        return {var: getattr(self, var) for var in all_vars}

    @staticmethod
    def from_dict(params_dict):
        """Create from dict."""
        params = BaseParams()
        for k, v in params_dict.items():
            setattr(params, k, v)
        return params
