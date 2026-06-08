# Bibliography



* **Towards the AlexNet Moment for HE** (2018) 
   * Proposed Homomorphic CNNs (HCNNs) and tackled challenges of adapting standard CNNs to the homomorphic domain, 
   * achieving 99% accuracy on MNIST and 77.55% on CIFAR-10, 
   * classifying a CIFAR-10 image in about 304 seconds.
* **Optimized Privacy-Preserving CNN Inference** — IEEE TIFS (2023) 
  * Proposed a more efficient way of evaluating convolutions under FHE where cost stays constant regardless of kernel size, 
  * achieving 12–46× timing improvement. 
  * Combined with bootstrapping, they achieved significant timing reductions on 20-layer CNNs evaluated on CIFAR-10/100 and ImageNet.
* **Toward Practical Privacy-Preserving CNNs** — arXiv 2023 
   * Proposed GPU/ASIC acceleration, efficient activation functions, 
   * optimized packing schemes for FHE-based CNN inference, 
   * reducing encrypted inference latency to 1.4 seconds on an NVIDIA A100 GPU for ResNet on CIFAR-10.
* HyPHEN — IEEE (2023) 
   * Introduced novel convolution algorithms and data packing methods 
   * to substantially reduce memory footprint and the number of expensive homomorphic operations like ciphertext rotation and bootstrapping in deep CNN inference under FHE.
* **Low-Complexity CNNs on FHE** — ICML 2022 
   * Introduced multiplexed parallel convolutions and imaginary-removing bootstrapping 
   * to efficiently run deep CNNs like ResNet under CKKS, 
   * addressing numerical divergence problems that arise with many layers.


* **CryptoNets** — Gilad-Bachrach et al., ICML 2016. First HE neural network inference; introduced polynomial activation approximations. (Solid — this one is well-established)


