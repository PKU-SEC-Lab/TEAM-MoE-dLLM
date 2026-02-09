# TEAM: Temporal–Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration

<div align="center">
  <img src="assets/1.png" alt="TBD" width="600">
</div>

We identify a fundamental mismatch between MoE architectures and dLLM. A large number of experts are activated at each denoising step, while only a small subset of tokens is ultimately accepted, resulting in substantial inference overhead and limiting their deployment in latency-sensitive applications. 

We propose TEAM, a plug-and-play framework that accelerates MoE dLLMs by enabling more accepted tokens with fewer activated experts. TEAM employs three complementary expert activation and decoding strategies, conservatively selecting necessary experts for decoded and masked tokens and simultaneously performing aggressive speculative exploration across multiple candidates.

**Overall Performance:**

With SDAR 30B-A3B ([SDAR](https://github.com/JetAstra/SDAR)) model, TEAM achieves an average speedup of 1.94× across diverse benchmarks, with a peak speedup of up to 2.2× on the HumanEval benchmark.
<div align="center">
  <img src="assets/Table1.png" alt="TBD" width="600">
</div>

### Installation
1.Clone the repository:
```
TBD
```
2.Install dependencies:
```
TBD
```

### Usage
TBD

Example:
```
TBD
```

### Acknowledgements
This repo is largely based on [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM). We would also like to thank the authors of [LLaDA](https://github.com/ML-GSAI/LLaDA) and [LLaDA-1.5](https://github.com/ML-GSAI/LLaDA-1.5) for their excellent work and open-source contributions.

### Contact
If you have any questions, please contact us via email lywei25@stu.pku.edu.cn.

