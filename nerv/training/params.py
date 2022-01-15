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
