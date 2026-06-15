import logging
import os
from typing import Optional

import h5py
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXP_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
REPO_DIR   = os.path.abspath(os.path.join(EXP_DIR, ".."))
BLD_DIR    = os.path.join(REPO_DIR, "build")
ESC50_ROOT       = os.path.join(REPO_DIR, "assets/esc-50/ESC-50-master")
ESC50_FPATH_META = os.path.join(ESC50_ROOT, "meta/esc50.csv")
ESC50_AUDIO_DIR = os.path.join(ESC50_ROOT, "audio")


LOG = logging.getLogger(__name__)
