import os

SCRIPT_DIR     = os.path.dirname(os.path.realpath(__file__))
REPO_DIR       = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
DPATH_ESC50    = os.path.join(REPO_DIR, "assets/esc-50/ESC-50-master")
FPATH_ESC_META = os.path.join(DPATH_ESC50, "meta/esc50.csv")


def test_esc_dir():
    assert os.path.isdir(DPATH_ESC50), f"ESC-50 dir not found: {DPATH_ESC50}"


def test_esc_meta_file():
    assert os.path.isfile(FPATH_ESC_META), f"ESC-50 metadata not found: {FPATH_ESC_META}"


def test_esc10_construct():
    from fhe_dsp.esc50 import Esc50Dataset
    ds = Esc50Dataset(esc50_metafile=FPATH_ESC_META, esc10=True)
    assert len(ds.classes) == 10
    assert all(isinstance(c, str) for c in ds.classes)


def test_esc50_construct():
    from fhe_dsp.esc50 import Esc50Dataset
    ds = Esc50Dataset(esc50_metafile=FPATH_ESC_META, esc10=False)
    assert len(ds.classes) == 50
    assert all(isinstance(c, str) for c in ds.classes)
    print(ds.classes)


def test_esc10_audio_files():
    from fhe_dsp.esc50 import Esc50Dataset
    audio_dir = os.path.join(DPATH_ESC50, "audio")
    ds = Esc50Dataset(esc50_metafile=FPATH_ESC_META, esc10=True)

    missing = [f for f in ds._df["filename"] if not os.path.isfile(os.path.join(audio_dir, f))]
    print(f"Audio files: {len(ds._df) - len(missing)}/{len(ds._df)}")
    assert not missing, f"Missing audio files:\n" + "\n".join(missing)


def test_esc50_audio_files():
    from fhe_dsp.esc50 import Esc50Dataset
    audio_dir = os.path.join(DPATH_ESC50, "audio")
    ds = Esc50Dataset(esc50_metafile=FPATH_ESC_META, esc10=False)
    missing = [f for f in ds._df["filename"] if not os.path.isfile(os.path.join(audio_dir, f))]
    print(f"Audio files: {len(ds._df) - len(missing)}/{len(ds._df)}")
    assert len(missing)==0, f"Failed to find {len(missing):,} audio files:"


def test_esc10_iter_audio():
    from fhe_dsp.esc50 import Esc50Dataset
    ds = Esc50Dataset(esc50_metafile=FPATH_ESC_META, esc10=True)
    paths = list(ds.iter_audio())
    assert len(paths) > 0, "No audio files yielded"
    assert all(p is not None for p in paths)
    assert all(os.path.isfile(p) for p in paths)
    print(f"Iterated {len(paths)} audio files")


def test_esc10_audio_dir_exists():
    from fhe_dsp.esc50 import Esc50Dataset
    ds = Esc50Dataset(esc50_metafile=FPATH_ESC_META, esc10=True)
    assert os.path.isdir(ds.audio_dir()), f"Audio dir not found: {ds.audio_dir()}"