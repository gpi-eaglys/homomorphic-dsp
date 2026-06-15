import logging
import os

from common import BLD_DIR, ESC50_FPATH_META
from fhe_dsp.train_mlp import MLP, train_all  # noqa: F401  (MLP re-exported for export_mdl)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    train_all(
        search_dir     = os.path.join(BLD_DIR, "fea"),
        mdl_root       = os.path.join(BLD_DIR, "mdl", "exp01"),
        esc50_metafile = ESC50_FPATH_META,
        esc10          = True,
        h5_prefix      = "esc10-",
        min_acc        = 0.95,
        patience       = 50,
    )
