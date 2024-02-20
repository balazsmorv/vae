import pytorch_lightning as pl
from functools import partial
import einops
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from Models.ldm import AutoencoderKL
import torch
import os
from datetime import datetime
from Data.dataloaders import ABIDELoader

pl.seed_everything(42, workers=True)

def main():
    model_ckpt = None
    batch_size = 2
    experiment = 'abide'
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_pth = f".{os.sep}logs"
    logger = TensorBoardLogger(save_dir=log_pth, name=experiment, version=time_stamp)
    with open(r"Configurations/ABIDE/vae_config.yml", 'r+') as yaml_file:
        pipeline_cfg = yaml.safe_load(yaml_file)

    transforms = {'fmri': [partial(einops.rearrange, pattern='b h l d -> b 1 d h l')]}
    datahandler = ABIDELoader(
        root_dir=r"/home/oem/Dokumentumok/ABIDE/data/Outputs/ccs/filt_noglobal/func_preproc",
        exp_path=r"./Configurations/ABIDE",
        transforms=transforms,
        batch_size=batch_size,
        num_workers=12,
        prefetch_factor=2,
        persistent_workers=True
    )
    datahandler.setup(stage='test')
    batch = next(iter(datahandler.test_dataloader()))

    model = AutoencoderKL(**pipeline_cfg)
    if model_ckpt is not None:
        vae_ckpt = torch.load(model_ckpt, map_location='cpu')
        model.load_state_dict(vae_ckpt['state_dict'])
        model.freeze()

    out = model.encoder(batch['fmri'])
    print(out)


if __name__ == '__main__':
    main()
