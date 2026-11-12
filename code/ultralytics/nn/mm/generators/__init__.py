# flake8: noqa
"""
模态生成器入口（对外暴露 DepthGen / DEMGen / EdgeGen / NormalGen）。
"""

from .depth_anything_v2 import DepthGen
from .dem_features import DEMGen
from .edge import EdgeGen
from .normal import NormalGen

__all__ = [
    "DepthGen",
    "DEMGen",
    "EdgeGen",
    "NormalGen",
]
