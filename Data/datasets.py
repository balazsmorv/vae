"""
Dataset handler for the pre-processed DS002336 (XP1) simultaneous EEG-fMRI dataset
"""
import os.path
from abc import abstractmethod
from os.path import join as opj
from typing import Literal, Tuple, List, Union

import pandas as pd
import nibabel as nib

import torch as pt
from torch.utils.data import Dataset

from Data.utils import std_norm_data_np, minmax_norm_data_np


class MedicalDatasetBase(Dataset):
    def __init__(
            self,
            root_dir: str,
            set_file: str,
            rescale: str = None
    ):
        super().__init__()

        self.root_dir = root_dir
        self.set_file = set_file
        self.rescale = rescale

        # Set in _load_set_info()
        self._set_info = None

        # Set in _encode_labels()
        self._class_ids = None
        self._encoded_labels = None

        # Set in _load_data_src()
        self._data_src = None

    def _setup(self):
        self._load_set_info()
        self._encode_labels()
        self._load_data_src()

    def __len__(self) -> int:
        return len(self._set_info.index)

    @abstractmethod
    def __getitem__(self, item: int) -> dict:
        raise NotImplementedError

    def _load_set_info(self) -> None:
        self._set_info = pd.read_csv(self.set_file, sep=';')

    @abstractmethod
    def _load_data_src(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _encode_labels(self) -> None:
        raise NotImplementedError

            
            
class ABIDEDataset(MedicalDatasetBase):
    def __init__(self, root_dir: str, set_file: str, rescale: str = None, cond_modality_list: List[str] = None):
        """
        Dataset handler for the pre-processed ABIDE I fMRI dataset.

        Args:
            root_dir: path of the directory containing data as a list of .nii.gz files
            set_file: path of a pre-define .csv file containing information for subject, label, and fmri matching
        """
        super().__init__(root_dir, set_file, rescale)
        self.cond_modality_list = cond_modality_list
        self._setup()

    def __getitem__(self, item: int) -> dict:
        data_info = self._set_info.iloc[item]
        try:
            data = {
                'label': self._encoded_labels[item],
                'fmri': self._data_src[data_info['FILE_ID']]['fmri'].dataobj[..., data_info['TIME_SLICE'].copy()],
                'subject': data_info['FILE_ID'],
                'affine': self._data_src[data_info['FILE_ID']]['fmri'].affine
            }
            
            if self.rescale is not None:
                stats = self._select_stats(data_info, self.rescale)
                data['fmri'] = self._rescale_data(data['fmri'], self.rescale, stats)
            
            if self.cond_modality_list is not None:
                context = self._get_condition_item(data_info)
                if self.rescale is not None:
                    stats = self._select_stats(data_info, self.rescale, 'ROI_')
                    # print("RESCALA #######################")
                    # print(stats)
                    context = self._rescale_data(context, self.rescale, stats)
                    # print(context.max(), context.min())
                    # exit()
                data.update({
                    'context': context
                })
        except EOFError:
            print("EOF Error, Check Data Info and regenerate labels!")
            print(data_info)
            raise
        return data

    def _get_condition_item(self, data_info: dict) -> Union[pt.Tensor, None]:
        if self.cond_modality_list is None:
            return None
        conditions = []
        for modality in self.cond_modality_list:
            mod_name = modality.split('.')[0]
            conditions.append(
                pt.from_numpy(self._data_src[data_info['FILE_ID']][mod_name].iloc[data_info['TIME_SLICE']].to_numpy()).float()
            )
        conditions = pt.concatenate(conditions, dim=0)
        return conditions

    def _load_data_src(self) -> None:
        sites_n_subjs = self._set_info['FILE_ID'].unique()
        self._data_src = {}
        for sns in sites_n_subjs:
            self._data_src[sns] = {'fmri': nib.load(opj(self.root_dir, f'{sns}_func_preproc.nii.gz'))}
            if self.cond_modality_list is not None:
                root_dir = os.path.split(self.root_dir)[0]
                for modality in self.cond_modality_list:
                    mod_name = modality.split('.')[0]
                    self._data_src[sns].update({
                        mod_name: pd.read_csv(opj(root_dir, mod_name, f'{sns}_{modality}'), sep='\t')
                    })

    def _encode_labels(self) -> None:
        self._class_ids = self._set_info['DX_GROUP'].unique()
        self._class_ids.sort()
        self._encoded_labels = [pt.tensor([1., 0.]).float() if label == 1 else pt.tensor([0., 1.]).float() 
                                for label in self._set_info['DX_GROUP'].to_numpy()]
        #one_hot(
        #    pt.from_numpy(
        #        self._set_info['DX_GROUP'].to_numpy() - self._class_ids[0]
        #    ).long(),
        #    num_classes=len(self._class_ids)
        #).float()
        
    def _select_stats(self, data, strategy, context_key=''):
        if 'std' in strategy:
            if 'subj' in strategy:
                return data[f'{context_key}SUBJ_AVG'], data[f'{context_key}SUBJ_STD']
            elif 'site' in strategy:
                return data[f'{context_key}SITE_AVG'], data[f'{context_key}SITE_STD']
            elif 'set' in strategy:
                return data[f'{context_key}SET_AVG'], data[f'{context_key}SET_STD']
            else:
                raise AttributeError(f'Not a valid stat! {strategy}. std and minmax can be used with subj, site and set')
        elif 'minmax' in strategy or 'unit' in strategy:
            if 'subj' in strategy:
                return data[f'{context_key}SUBJ_MIN'], data[f'{context_key}SUBJ_MAX']
            elif 'site' in strategy:
                return data[f'{context_key}SITE_MIN'], data[f'{context_key}SITE_MAX']
            elif 'set' in strategy:
                return data[f'{context_key}SET_MIN'], data[f'{context_key}SET_MAX']
            else:
                raise AttributeError(f'Not a valid stat! {strategy}. std and minmax can be used with subj, site and set')
        else:
            raise AttributeError(f'Not a valid stat! {strategy}. std and minmax can be used with subj, site and set')
        
    def _rescale_data(self, data: pt.Tensor, strategy: str, stats: tuple = None) -> pt.Tensor:
        if "std_norm" in strategy:
            return std_norm_data_np(data, stats=stats, dim=0)
        elif "unit_norm" in strategy or "minmax_norm" in strategy:
            if  "minmax_norm" in strategy:
                norm_range = (-1., 1.)
            else:
                norm_range = (0., 1.)
            return minmax_norm_data_np(data, stats=stats, norm_range=norm_range, dim=0) 


class ABIDEDataset4D(ABIDEDataset):
    def __init__(self, root_dir: str, set_file: str, rescale: str = None, cond_modality_list: List[str] = None):
        """
        Dataset handler for the pre-processed ABIDE I fMRI dataset;  4D data

        Args:
            root_dir: path of the directory containing data as a list of .nii.gz files
            set_file: path of a pre-define .csv file containing information for subject, label, and fmri matching
        """
        super().__init__(root_dir, set_file, rescale)
        self.cond_modality_list = cond_modality_list
        self._setup()

    def __getitem__(self, item: int) -> dict:
        data_info = self._set_info.iloc[item]
        try:
            data = {
                'label': self._encoded_labels[item],
                'fmri': self._data_src[data_info['FILE_ID']]['fmri'].dataobj[..., data_info['START_TIME']:data_info['END_TIME']].copy(),
                'subject': f"{data_info['FILE_ID']}_{data_info['START_TIME']}_{data_info['END_TIME']}",
                'affine': self._data_src[data_info['FILE_ID']]['fmri'].affine
            }
            if self.rescale is not None:
                stats = self._select_stats(data_info, self.rescale)
                data['fmri'] = self._rescale_data(data['fmri'], self.rescale, stats)
            
            if self.cond_modality_list is not None:
                context = self._get_condition_item(data_info, data_info['START_TIME'], data_info['END_TIME'])
                if self.rescale is not None:
                    stats = self._select_stats(data_info, self.rescale, 'ROI_')
                    context = self._rescale_data(context, self.rescale, stats)
                data.update({
                    'context': context
                })
        except EOFError:
            print("EOF Error, Check Data Info and regenerate labels!")
            print(data_info)
            raise
        return data

    def _get_condition_item(self, data_info: dict, start_time: int, stop_time: int) -> Union[pt.Tensor, None]:
        if self.cond_modality_list is None:
            return None
        conditions = []
        for modality in self.cond_modality_list:
            mod_name = modality.split('.')[0]
            conditions.append(
                pt.from_numpy(self._data_src[data_info['FILE_ID']][mod_name].iloc[start_time:stop_time].to_numpy()).float()
            )
        conditions = pt.concatenate(conditions, dim=0)
        return conditions


class WhiteNoiseDataset(Dataset):
    def __init__(self, num_samples: int, shape: Tuple[int, ...]):
        super().__init__()
        self.num_samples = num_samples
        self.shape = shape
        self.data = pt.randn((self.num_samples, *shape))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, item: int) -> dict:
        return {'fmri': self.data[item]}
