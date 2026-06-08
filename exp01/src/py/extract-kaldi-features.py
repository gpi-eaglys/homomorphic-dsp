import os
import logging
from collections.abc import Generator

import h5py
import torchaudio
import kaldifeat
import polars as pl

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
BLD_DIR = os.path.join(EXP_DIR, "build")
REPO_DIR = os.path.abspath(os.path.join(EXP_DIR, ".."))
# download META_CSV with 'exp01/scripts/download-dataset.sh' && extract
META_CSV = os.path.join(REPO_DIR, "assets/esc-50/ESC-50-master/meta/esc50.csv")
AUDIO_DIR = os.path.join(REPO_DIR, "assets/esc-50/ESC-50-master/audio")


LOG = logging.getLogger(__name__)


def iter_esc10() -> Generator[str, None, None]:
    """
    Yield absolute paths to audio files in the ESC-10 sub-category of ESC-50. Skips missing files.
    """
    df = pl.read_csv(META_CSV).filter(pl.col("esc10"))
    LOG.debug("Loaded %d esc10 rows", len(df))
    for filename in df["filename"]:
        path = os.path.join(AUDIO_DIR, filename)
        if os.path.isfile(path):
            yield path
        else:
            LOG.warning("Cannot find audio: %s", path)



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
    Does feature extraction for "esc10" subset.
    """
    dpath_feat = os.path.join(BLD_DIR, "fea")
    fpath_feat = os.path.join(dpath_feat, f"esc10-{feat}.h5")

    if os.path.isfile(fpath_feat):
        LOG.info(f"Skipping extraction: found feature file: {os.path.relpath(fpath_feat, REPO_DIR)}.")
        return
    os.makedirs(dpath_feat, exist_ok=True)

    fe = get_fe(feat)

    with h5py.File(fpath_feat, "w") as f:
        for i, fpath_wav in enumerate(iter_esc10()):
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
    LOG.debug("Loading metadata from %s", META_CSV)
    do_feat_ext("mfb")
    # do_feat_ext("mfcc")

