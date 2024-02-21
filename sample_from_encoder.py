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

pl.seed_everything(42, workers=True)

def main():
    model_ckpt = "/Users/balazsmorvay/Downloads/epoch=3-step=4500.ckpt"
    batch_size = 1
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    asset_path = os.path.join('Assets/', time_stamp)
    os.mkdir(asset_path)
    with open(r"Configurations/ABIDE/vae_config.yml", 'r+') as yaml_file:
        pipeline_cfg = yaml.safe_load(yaml_file)

    labels = pd.DataFrame(columns=['FILE_ID', 'DX_GROUP'])

    transforms = {'fmri': [partial(einops.rearrange, pattern='b h l d -> b 1 d h l')]}
    datahandler = ABIDELoader(
        root_dir=r"/Users/balazsmorvay/Downloads/ABIDE/data/Outputs/ccs/filt_noglobal/func_preproc",
        exp_path=r"./Configurations/ABIDE",
        transforms=transforms,
        batch_size=batch_size,
        num_workers=12,
        prefetch_factor=2,
        persistent_workers=True
    )
    datahandler.setup(stage='test', test_filename='NYU_all.csv')
    dataloader = datahandler.test_dataloader()

    model = AutoencoderKL(**pipeline_cfg)
    if model_ckpt is not None:
        vae_ckpt = torch.load(model_ckpt, map_location='cpu')
        model.load_state_dict(vae_ckpt['state_dict'])
        model.freeze()
    encoder = model.encoder.to(device='cuda' if torch.cuda.is_available() else 'cpu').eval()

    with torch.inference_mode():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            inp = batch['fmri']
            label = np.array(batch['label'].to('cpu'))[0]
            file_id = batch['file_id'][0]
            out = np.array(encoder(inp).to('cpu'))
            np.save(file=os.path.join(asset_path, f'{file_id}.npy'), arr=out)
            labels.loc[len(labels)] = [file_id, 1 if label[0] == 1.0 else 2]

    labels.to_csv(path_or_buf=os.path.join(asset_path, 'labels.csv'))


if __name__ == '__main__':
    main()
