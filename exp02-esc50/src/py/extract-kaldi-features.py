import os
import logging

import h5py
import torchaudio
import kaldifeat

from common import REPO_DIR, ESC50_FPATH_META

BLD_DIR = os.path.join(REPO_DIR, "build")
from fhe_dsp.esc50 import Esc50Dataset


LOG = logging.getLogger(__name__)


def _get_fe_for_mfb(nbins: int = 40):
    """
    Use default mel filter bank
    cf. 
    compute-fbank-feats --dither=0 scp:test.scp ark,t:test.txt
    """
    # opts.device = torch.device("cuda", 0)
    # features = fbank(wave.to(opts.device))

    opts = kaldifeat.FbankOptions()
    opts.frame_opts.dither = 0    
    opts.mel_opts.num_bins = nbins  # 80
    LOG.debug("MFB options: %s", opts)
    fbank = kaldifeat.Fbank(opts)    
    return fbank

def _get_fe_for_mfcc(nceps: int = 20):
    opts = kaldifeat.MfccOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 2*nceps
    opts.num_ceps = nceps
    LOG.debug("MFCC options: %s", opts)
    mfcc = kaldifeat.Mfcc(opts) 
    return mfcc


def get_fe(feat: str):
    match feat:
        case "mfb" | "fbank":
            return _get_fe_for_mfb(40)
        case "mfcc":
            return _get_fe_for_mfcc(40)
        case _:
            raise ValueError(f"Unknown feature type: {feat}")


def do_feat_ext(feat = "mfb"):
    """
    Does feature extraction for the whole "esc50" subset.
    """
    dpath_feat = os.path.join(BLD_DIR, "fea")
    fpath_feat = os.path.join(dpath_feat, f"esc50-{feat}.h5")

    if os.path.isfile(fpath_feat):
        LOG.info(f"Skipping extraction: found feature file: {os.path.relpath(fpath_feat, REPO_DIR)}.")
        return
    os.makedirs(dpath_feat, exist_ok=True)

    esc50 = Esc50Dataset(ESC50_FPATH_META, esc10=False)

    fe = get_fe(feat)

    with h5py.File(fpath_feat, "w") as f:
        for i, fpath_wav in enumerate(esc50.iter_audio()):
            key = os.path.splitext(os.path.basename(fpath_wav))[0]
            wave, hz = torchaudio.load(fpath_wav)
            wave = wave.squeeze()
            wave *= 32768  # for kaldi compatibility
            features = fe(wave).numpy()
            f.create_dataset(key, data=features)
            LOG.debug("Wrote (%4d) %-14s: shape=%s", i+1, key, features.shape)

    LOG.info("Saved %d features to: %s", i+1, os.path.relpath(fpath_feat, REPO_DIR))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    LOG.debug("Loading metadata from %s", ESC50_FPATH_META)
    do_feat_ext("mfb")
    do_feat_ext("mfcc")


