class BaseParams:
    # training settings
    gpus = 1
    max_epochs = 100
    san_check_val_step = 1
    print_iter = 50
    save_interval = 1.0

    # optimizer
    lr = 1e-3
    weight_decay = 0.0
    clip_grad = -1

    # data
    data_root = ""
    train_batch_size = 64
    val_batch_size = 64
    num_workers = 4
