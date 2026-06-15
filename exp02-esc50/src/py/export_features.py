import logging
import os

from common import BLD_DIR
from fhe_dsp.export_features import export_all

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    export_all(os.path.join(BLD_DIR, "fea"))
