import os


def find_old_slurm_id(ckp_path):
    """Find the SLURM_JOB_ID from the last crashed run."""
    # we will have a `${ckp_path}/wandb/run-YYYYMMDD_TIME-xxx-${OLD_ID}` dir
    # retrieve the old id from it
    wandb_dir = os.path.join(ckp_path, 'wandb')
    if not os.path.exists(wandb_dir):
        return None
    for f in os.listdir(os.path.join(wandb_dir)):
        # YYYYMMDD, update in case we are still using this code in 2100!
        if f.startswith('run-20'):
            return f.split('-')[-1]
    return None
