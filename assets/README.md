

## ESC-50

2000 labeled environmental audio clips (5 s each, 44.1 kHz), 50 classes, 40 clips per class.

### meta/esc50.csv columns

| Column | Description |
|---|---|
| `filename` | Audio file name, encodes `{fold}-{src_file}-{take}-{target}.wav` |
| `fold` | Cross-validation fold (1–5). Use for reproducible train/test splits. |
| `target` | Numeric class label (0–49) |
| `category` | Human-readable class name (e.g. `dog`, `rain`) |
| `esc10` | `True` if the clip belongs to the ESC-10 subset (10 easiest classes) |
| `src_file` | Freesound recording ID the clip was cut from |
| `take` | Take letter (A, B, C…) — distinguishes multiple clips from the same source recording |

The filename is redundant with the CSV columns but lets you identify a clip without the metadata file.
