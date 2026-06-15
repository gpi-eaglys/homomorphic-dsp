import logging
import os

from common import BLD_DIR, ESC50_FPATH_META
from fhe_dsp.export_mdl import export_all
from fhe_dsp.train_mlp import MLP

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    export_all(
        mdl_root       = os.path.join(BLD_DIR, "mdl", "exp02"),
        feat_root      = os.path.join(BLD_DIR, "fea"),
        esc50_metafile = ESC50_FPATH_META,
        esc10          = False,
        mlp_cls        = MLP,
    )
