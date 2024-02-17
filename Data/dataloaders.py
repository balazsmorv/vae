import os.path

# Python libraries
import yaml
from os.path import join as opj
import torch as pt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.transforms as trafo

from typing import Dict


# Project imports
from Data.utils import std_norm_data_pt, minmax_norm_data_pt
from Data.datasets import *


class MedicalDataLoaderBase(pl.LightningDataModule):
    def __init__(
            self,
            root_dir: str,
            exp_path: str,
            batch_size: int,
            transforms: Dict[str, list] = None,
            rescale: bool = False,
            num_workers: int = 0,
            prefetch_factor: int = None,
            persistent_workers: bool = None
    ):
        super().__init__()
        assert os.path.exists(root_dir)

        self.root_dir = root_dir
        self.exp_path = exp_path
        self.batch_size = batch_size
        self.transforms = transforms
        self.rescale = rescale
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        with open(opj(self.exp_path, 'config.yaml'), 'r+') as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        # Set in setup()
        self.train_set = None
        self.test_set = None
        self.val_set = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)

    def _rescale_data(self, data: pt.Tensor, strategy: str) -> pt.Tensor:
        if strategy == "std_norm":
            return std_norm_data_pt(data, dim=(1, 2, 3))
        elif strategy == "unit_norm" or strategy == "minmax_norm":
            if strategy == "minmax_norm":
                norm_range = (-1., 1.)
            else:
                norm_range = (0., 1.)
            return minmax_norm_data_pt(data, norm_range=norm_range, dim=(1, 2, 3)) 
        
    def _resize_data(self, data: pt.Tensor, strategy: int) -> pt.Tensor:
        data = data.permute(0,3,1,2)
        b,c,h,w = data.shape
        
        # for some interesting reason F.pad does not work like it is supposed, that is F.pad(data, (0,1,0,4)) should pad the last dim with (0,1), and the second to last with (0,4), but here it is reversed,         (0,1) is for the second to last
        transforms = trafo.Compose([
            #trafo.Lambda(lambda x: x[:,:,1:,1:]),
            #trafo.Lambda(lambda x: F.pad(x, (0, self.config['fmri_settings']['resize'][0] - (h-1), self.config['fmri_settings']['resize'][1] - (w-1), 0))),
            trafo.Resize(size = (self.config['fmri_settings']['resize'][0] * 2, self.config['fmri_settings']['resize'][1] * 2))
        ])
        
        data = transforms(data)
        return data


class DS002336Loader(MedicalDataLoaderBase):
    def __init__(
            self,
            root_dir: str,
            exp_path: str,
            batch_size: int,
            transforms: Dict[str, list] = None,
            rescale: bool = False,
            num_workers: int = 0,
            prefetch_factor: int = None,
            persistent_workers: bool = None
    ):
        """
        Handler class for the DS002336 (XP1) simultaneous EEG-fMRI dataset.

        Args:
            root_dir: path of the directory containing data in BIDS format
            exp_path: path of the directory containing subset configuration files in .csv format
            transforms: list of transformations as partial functions to be applied on the data
            rescale: whether to rescale (e.g. normalize, standardize) the data
        """
        super().__init__(root_dir, exp_path, batch_size, transforms, rescale,
                         num_workers, prefetch_factor, persistent_workers)

    def setup(self, stage: str) -> None:
        # Load data
        if stage == 'fit':
            self.train_set = DS002336(self.root_dir, opj(self.exp_path, 'train.csv'), 
                                      modality=self.config['modality'],
                                      fmri_cleaned=self.config['fmri_settings']['cleaned'],
                                      eeg_prefix=self.config['eeg_settings']['prefix'],
                                      eeg_range=self.config['eeg_settings']['range'])
            try:
                self.val_set = DS002336(self.root_dir, opj(self.exp_path, 'val.csv'),
                                        modality=self.config['modality'],
                                        fmri_cleaned=self.config['fmri_settings']['cleaned'],
                                        eeg_prefix=self.config['eeg_settings']['prefix'],
                                        eeg_range=self.config['eeg_settings']['range'])
            except:
                print("NO VAL FOR THIS SET!")

        if stage == 'test':
            self.test_set = DS002336(self.root_dir, opj(self.exp_path, 'test.csv'),
                                     modality=self.config['modality'],
                                     fmri_cleaned=self.config['fmri_settings']['cleaned'],
                                     eeg_prefix=self.config['eeg_settings']['prefix'],
                                     eeg_range=self.config['eeg_settings']['range'])

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        # Rescale data samples
        if ('eeg' in self.config['modality']) and (self.config['eeg_settings']['rescale'] is not None):
            batch['eeg'] = self._rescale_data(batch['eeg'], strategy=self.config['eeg_settings']['rescale'])
        if ('fmri' in self.config['modality']) and (self.config['fmri_settings']['rescale'] is not None):
            batch['fmri'] = self._rescale_data(batch['fmri'], strategy=self.config['fmri_settings']['rescale'])
            # if (self.config['fmri_settings']['resize'] is not None):
            #     batch['fmri'] = self._resize_data(batch['fmri'], strategy=self.config['fmri_settings']['resize'])

        if self.transforms is not None:
            if ('fmri' in self.config['modality']) and ('fmri' in self.transforms.keys()):
                for transform in self.transforms['fmri']:
                    batch['fmri'] = transform(batch['fmri'])
                batch['fmri'] = batch['fmri'].contiguous()
        return batch


class ABIDELoader(MedicalDataLoaderBase):
    def __init__(
            self,
            root_dir: str,
            exp_path: str,
            batch_size: int,
            transforms: Dict[str, list] = None,
            rescale: bool = False,
            num_workers: int = 0,
            prefetch_factor: int = None,
            persistent_workers: bool = None,
    ):
        """
        Handler class for the  ABIDE I pre-processed fMRI dataset.

        Args:
            root_dir: path of the directory containing data as a list of .nii.gz files
            exp_path: path of the directory containing subset configuration files in .csv format
            transforms: list of transformations as partial functions to be applied on the data
            rescale: whether to rescale (e.g. normalize, standardize) the data
        """
        super().__init__(root_dir, exp_path, batch_size, transforms, rescale,
                         num_workers, prefetch_factor, persistent_workers)

    def setup(self, stage: str) -> None:
        dataset = ABIDEDataset
        # Load data
        if stage == 'fit':
            self.train_set = dataset(self.root_dir, opj(self.exp_path, 'train.csv'), self.config['rescale'], self.config['modalities'])
            self.val_set = dataset(self.root_dir, opj(self.exp_path, 'val.csv'), self.config['rescale'], self.config['modalities'])
        elif stage == 'test':
            self.test_set = dataset(self.root_dir, opj(self.exp_path, 'test.csv'), self.config['rescale'], self.config['modalities'])
        elif stage == 'pred4train':
            self.test_set = dataset(self.root_dir, opj(self.exp_path, 'train.csv'), self.config['rescale'], self.config['modalities'])
        elif stage == 'predict':
            print(self.test_set.set_file)
        else:
            raise AttributeError(f'Wrong stage was given! ({stage}) Possible: fit, test, pred4train')

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        if self.transforms is not None:
            for key in self.transforms.keys():
                for transform in self.transforms[key]:
                    batch[key] = transform(batch[key])
                batch[key] = batch[key].contiguous()
        return batch


class HCPLoader(MedicalDataLoaderBase):
    def __init__(
            self,
            root_dir: str,
            exp_path: str,
            batch_size: int,
            transforms: Dict[str, list] = None,
            rescale: bool = False,
            num_workers: int = 0,
            prefetch_factor: int = None,
            persistent_workers: bool = None
    ):
        """
        Handler class for the HCP pre-processed fMRI dataset.

        Args:
            root_dir: path of the directory containing data as a list of .nii.gz files
            exp_path: path of the directory containing subset configuration files in .csv format
            transforms: list of transformations as partial functions to be applied on the data
            rescale: whether to rescale (e.g. normalize, standardize) the data
        """
        super().__init__(root_dir, exp_path, batch_size, transforms, rescale,
                         num_workers, prefetch_factor, persistent_workers)

    def setup(self, stage: str) -> None:
        dataset = HCPDataset4D if ('length' in self.config.keys() and self.config['length'] > 1) else HCPDataset
        # Load data
        if stage == 'fit':
            self.train_set = dataset(self.root_dir, opj(self.exp_path, 'train.csv'), self.config['rescale'], self.config['modalities'])
            self.val_set = dataset(self.root_dir, opj(self.exp_path, 'val.csv'), self.config['rescale'], self.config['modalities'])
        elif stage == 'test':
            self.test_set = dataset(self.root_dir, opj(self.exp_path, 'test.csv'), self.config['rescale'], self.config['modalities'])
        elif stage == 'pred4train':
            self.test_set = dataset(self.root_dir, opj(self.exp_path, 'train.csv'), self.config['rescale'], self.config['modalities'])
        elif stage == 'predict':
            print(self.test_set.set_file)
        else:
            raise AttributeError(f'Wrong stage was given! ({stage}) Possible: fit, test, pred4train')

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        if self.transforms is not None:
            for key in self.transforms.keys():
                for transform in self.transforms[key]:
                    batch[key] = transform(batch[key])
                batch[key] = batch[key].contiguous()
        return batch
