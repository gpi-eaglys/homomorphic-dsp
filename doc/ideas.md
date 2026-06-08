

## TurboQuant 
* https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
* directly not related to quantization 
  * mainly for KV store in LLMs



## CKKS quantization with delta
* during scaling of input (delta multiplication)
* interpolate the input values
* according to some LUT or function





# Features 

## Kaldi features 
* usual suspects MFB, MFC, MFCC 
* with a bit higher dimensions 



## 


### audio noise detection

For real-time identification of general environmental sounds (sound event detection / audio tagging), here's the best fit given your constraints.

**Top recommendation: YAMNet**

YAMNet is the strongest practical choice for your case. It's trained on AudioSet, classifies into 521 sound event classes (dog bark, glass breaking, alarm, speech, etc.), and is specifically designed to be lightweight and fast — it runs comfortably in real time on CPU, including in streaming setups. It processes audio in short frames (~0.96 s windows), which fits a streaming pipeline naturally. Weights are freely downloadable, so it runs fully on-prem. It's available in TensorFlow and as TFLite, and converts cleanly to ONNX for C++ deployment via ONNX Runtime — which matters given your Python + C++ requirement.

**Strong alternatives**

- **PANNs (CNN14)** — Higher accuracy than YAMNet on AudioSet tagging and the same 527-class space, but heavier. It can still run in real time on a GPU, or on CPU with a slightly larger window/latency budget. Choose this if accuracy matters more than minimal footprint. PyTorch, exports to ONNX.
- **PaSST / BEATs** — Best-in-class accuracy, but transformer-based and harder to push to true low-latency streaming on CPU. Better for offline/batch or when you have GPU headroom. Probably overkill for real-time on the edge.
- **AST (Audio Spectrogram Transformer)** — Similar tradeoff: excellent accuracy, heavier inference.

How I'd decide:

- Real-time on CPU / edge, minimal footprint, easy C++ export → **YAMNet**
- Real-time but you have a GPU and want higher accuracy → **PANNs (CNN14)**
- You can tolerate batch/offline for top accuracy → **BEATs** or **PaSST**

Given everything from this thread (Python + C++ shared deployment, on-prem, now real-time), YAMNet is the cleanest fit: small, streaming-friendly, downloadable, and ONNX-exportable so a single model serves both your Python and C++ paths.

Two practical notes for a streaming setup. First, AudioSet-trained models give you general classes out of the box, but if your "noises" are a specific set (e.g., just glass-break + alarm + dog), you'll get much better precision by using the model as a feature extractor and training a small classifier head on your own labeled clips. Second, for real-time you'll typically run a sliding window with overlap and smooth predictions over a few frames to avoid jittery detections.

Do you need to detect a fixed, known set of sounds, or open-ended "what is this"? If it's a fixed set, I can sketch the fine-tuning/streaming approach concretely.

### YAMNet is crap. 

Fair enough — YAMNet's general AudioSet tagging is too coarse for reliably picking out specific environmental sources like helicopters. A few directions that fit better:

**Use a stronger backbone, then specialize it**

The realistic path to good helicopter/aircraft detection isn't an off-the-shelf classifier — it's a strong embedding model plus a small head trained on your target classes:

- **BEATs** or **PaSST** as the feature extractor — both substantially outperform YAMNet on AudioSet and produce much more discriminative embeddings for environmental sources. AudioSet already contains relevant classes (helicopter, aircraft, fixed-wing aircraft, propeller/airscrew), so these models have seen this kind of signal during training. You'd extract embeddings and train a lightweight classifier on your own labeled helicopter/non-helicopter clips. This gives far better precision than generic tagging.
- **PANNs (CNN14)** — same approach, lighter weight, still real-time capable on GPU and CPU-feasible. Good middle ground if BEATs/PaSST are too heavy for your latency budget.

**Why fine-tuning matters here**

Helicopters have a very characteristic acoustic signature — strong low-frequency periodicity from rotor blade-pass frequency and its harmonics, plus broadband modulation. A small classifier trained on even a few hundred labeled clips will dramatically outperform any generic tagger, because you're tuning the decision boundary to exactly the sources you care about and your specific recording conditions (mic, distance, background). This is the single biggest quality lever.

**Domain-specific angle**

Detecting aircraft/rotorcraft by sound overlaps with acoustic surveillance and UAV/drone detection research, where this is a well-studied problem. The standard recipe there is exactly the above: log-mel or CQT features (CQT is nice for capturing harmonic rotor structure), a CNN or CRNN, trained on curated positive/negative sets. If you want, the periodicity of the rotor also lends itself to explicit harmonic/cepstral features as an auxiliary signal.

How I'd approach it concretely:

1. Pull positive examples — AudioSet has a "Helicopter" class with thousands of clips you can download segments for; supplement with your own recordings in-environment.
2. Gather hard negatives — other aircraft, wind, traffic, machinery, anything in your deployment soundscape.
3. Extract embeddings with **BEATs** (best quality) or **PANNs** (lighter), or train a small CRNN on log-mel/CQT directly.
4. Train a small classifier head, run it streaming with a sliding window and temporal smoothing.

All of this runs on-prem, and the backbone exports to ONNX for your C++ path, consistent with the rest of your stack.

A couple of questions that would let me get specific: is this a fixed known set of targets (helicopter, maybe a few other aircraft) or open-ended? And do you have any labeled audio from the actual deployment environment yet, or would you be starting from public datasets? That determines whether this is mostly a data-collection task or mostly a modeling task.