import pytorch_lightning as pl
from functools import partial
import einops
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from Models.ldm import AutoencoderKL
import torch
import numpy as np
import os
import pandas as pd
from datetime import datetime
from Data.dataloaders import ABIDELoader
from tqdm import tqdm
import torch.nn.functional as F

pl.seed_everything(42, workers=True)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def main():
    model_ckpt = "/home/oem/Dokumentumok/vae/logs/abide_flat_z/240427_102110abideNYU/epoch=909-step=862000.ckpt"
    batch_size = 1
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    asset_path = os.path.join('Assets/', time_stamp)
    os.mkdir(asset_path)
    with open(r"Configurations/ABIDE/vae_config.yml", 'r+') as yaml_file:
        pipeline_cfg = yaml.safe_load(yaml_file)

    labels = pd.DataFrame(columns=['FILE_ID', 'DX_GROUP', 'TIME_SLICE'])

    transforms = {'fmri': [partial(einops.rearrange, pattern='b h l d -> b 1 d h l'),
                           partial(F.pad, pad=(0, 0, 3, 3))]}

    datahandler = ABIDELoader(
        root_dir=r"/home/oem/Dokumentumok/ABIDE/data/Outputs/ccs/filt_noglobal/func_preproc",
        exp_path=r"./Configurations/ABIDE",
        transforms=transforms,
        batch_size=batch_size,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=None,
        rescale=True
    )
    datahandler.setup(stage='test', test_filename='NYU_all.csv')
    dataloader = datahandler.test_dataloader()

    model = AutoencoderKL(**pipeline_cfg)
    if model_ckpt is not None:
        vae_ckpt = torch.load(model_ckpt, map_location='cpu')
        model.load_state_dict(vae_ckpt['state_dict'])
        model.freeze()
        model.eval()

    with torch.inference_mode():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            inp = datahandler.on_after_batch_transfer(batch, idx)
            inp_fmri = model.get_input(inp, 'fmri')
            label = np.array(batch['label'].to('cpu'))[0]
            file_id = batch['subject'][0]
            time_slice = int(batch['time_slice'][0])
            latent = np.array(model.encode(inp_fmri.squeeze(0)).mode().to('cpu'))
            latent = (latent - latent.min()) / (latent.max() - latent.min()) * 2 - 1
            np.save(file=os.path.join(asset_path, f'{file_id}.npy'), arr=latent)
            labels.loc[len(labels)] = [file_id, 1 if label[0] == 1.0 else 2, time_slice]

    labels.to_csv(path_or_buf=os.path.join(asset_path, 'labels.csv'))


if __name__ == '__main__':
    main()
