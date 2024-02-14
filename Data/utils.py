
import torch as pt
import numpy as np
from typing import Union, Tuple


def std_norm_data_pt(
        data: pt.Tensor,
        stats: Tuple[float, float] = None,
        dim: Union[int, tuple] = 0,
        eps: float = 1e-7
) -> pt.Tensor:
    std, mean = (stats[1], stats[0]) if stats is not None else pt.std_mean(data, dim=dim, keepdim=True)
    return (data - mean) / (std + eps)


def minmax_norm_data_pt(
        data: pt.Tensor,
        norm_range: Tuple[float, float] = (-1.0, 1.0),
        dim: Union[int, tuple] = 0,
        stats: Tuple[float, float] = None,
        eps: float = 1e-7
) -> pt.Tensor:
    stats = stats if stats is not None else (data.amin(dim, keepdim=True), data.amax(dim, keepdim=True))
    return norm_range[0] + ((data - stats[0]) * (norm_range[1] - norm_range[0])) / (stats[1] - stats[0] + eps)


def std_norm_data_np(
        data: np.ndarray,
        stats: Tuple[float, float] = None,
        dim: Union[int, tuple] = 0,
        eps: float = 1e-7
) -> np.ndarray:
    if stats is not None:
        mean, std = stats
    else:
        mean, std = data.mean(axis=dim, keepdims=True), np.std(data, axis=dim, keepdims=True)
    return (data - mean) / std


def minmax_norm_data_np(
        data: np.ndarray,
        norm_range: Tuple[float, float] = (-1.0, 1.0),
        dim: Union[int, tuple] = 0,
        stats: Tuple[float, float] = None,
        eps: float = 1e-7
) -> np.ndarray:
    stats = stats if stats is not None else (data.min(axis=dim, keepdims=True), data.max(axis=dim, keepdims=True))
    diff = stats[1] - stats[0]
    diff = eps if np.abs(diff) < eps else diff
    return norm_range[0] + ((data - stats[0]) * (norm_range[1] - norm_range[0])) / diff


def renorm(
        data: pt.Tensor
) -> pt.Tensor:
    return (data + 1) * 0.5

