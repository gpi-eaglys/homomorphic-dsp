# Experiment 01: Urban noises 
* The goal is to classify urban noises homomorphemically

# TL;DR 
* download audio 
```
./scripts/download-dataset.sh 
```

* extract features (MFB and MFCC)
```
uv run src/py/extract-kaldi-features.py
```

* train MLP
```
uv run --package exp01  python exp01-esc10/src/py/train_mlp.py
```




## Data-U: Urban Sound
* https://urbansounddataset.weebly.com/urbansound8k.html
* 8732 audio files
* 10 classes


## Data-E: ESC-50
* ESC-50: https://github.com/karolpiczak/ESC-50
* direct data link: https://github.com/karoldvl/ESC-50/archive/master.zip
* kaggle: https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50
* 2000 audio files, 5 seconds each
* 50 classes





