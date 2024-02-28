import pytorch_lightning as pl
from functools import partial
import einops
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from Models.ldm import AutoencoderKL
import torch
from torch.nn import functional as F
import os
from datetime import datetime
from Data.dataloaders import ABIDELoader

pl.seed_everything(42, workers=True)

def main():
    model_ckpt = None
    experiment_name = 'abide_video'
    site_used = "NYU"
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S") + 'abide' + site_used
    log_pth = f".{os.sep}logs"
    logger = TensorBoardLogger(save_dir=log_pth, name=experiment_name, version=time_stamp)
    with open(r"Configurations/ABIDE/vae_config.yml", 'r+') as yaml_file:
        pipeline_cfg = yaml.safe_load(yaml_file)

    batch_size = 32
    transforms = {
        'fmri': [
            partial(einops.rearrange, pattern='b h l d -> b d h l'),
            partial(F.pad, pad=(0, 0, 3, 3))
        ]
    }

    datahandler = ABIDELoader(
        root_dir=r"/home/oem/Dokumentumok/ABIDE/data/Outputs/ccs/filt_noglobal/func_preproc",
        exp_path=r"./Configurations/ABIDE_video",
        transforms=transforms,
        batch_size=batch_size,
        rescale=True,
        num_workers=12,
        prefetch_factor=2,
        persistent_workers=None
    )

    model = AutoencoderKL(**pipeline_cfg)
    if model_ckpt is not None:
        vae_ckpt = torch.load(model_ckpt, map_location='cpu')
        model.load_state_dict(vae_ckpt['state_dict'])

    ckpt_callback = ModelCheckpoint(dirpath=logger.log_dir, every_n_train_steps=750)
    early_stopping_callback = EarlyStopping(monitor='val/total_loss', mode='min', patience=12, strict=True)
    trainer = pl.Trainer(logger=logger,
                         precision='32',
                         val_check_interval=0.25,
                         limit_val_batches=0.15,
                         #gradient_clip_val=1.,
                         detect_anomaly=True,
                         callbacks=[ckpt_callback, early_stopping_callback],
                         inference_mode=True)
    trainer.fit(model=model, datamodule=datahandler)


if __name__ == '__main__':
    main()