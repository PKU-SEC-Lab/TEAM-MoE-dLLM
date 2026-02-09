<img width="1533" height="1014" alt="image" src="https://github.com/user-attachments/assets/e3b975e1-a008-428e-8804-6a3cb326208d" /># TEAM: Temporal–Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration

<div align="center">
  <img src="assets/1.png" alt="TBD" width="500">
</div>

We identify a fundamental mismatch between MoE architectures and dLLM. A large number of experts are activated at each denoising step, while only a small subset of tokens is ultimately accepted, resulting in substantial inference overhead and limiting their deployment in latency-sensitive applications. 

We propose TEAM, a plug-and-play framework that accelerates MoE dLLMs by enabling more accepted tokens with fewer activated experts. TEAM employs three complementary expert activation and decoding strategies, conservatively selecting necessary experts for decoded and masked tokens and simultaneously performing aggressive speculative exploration across multiple candidates.

**Overall Performance:**

With SDAR 30B-A3B ([SDAR](https://github.com/JetAstra/SDAR)) model, TEAM achieves an average speedup of 1.94× across diverse benchmarks, with a peak speedup of up to 2.2× on the HumanEval benchmark.
<div align="center">
  <img src="assets/Table1.png" alt="TBD" width="900">
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
This repo is largely based on [SDAR](https://github.com/JetAstra/SDAR). We would like to thank the authors of this for their excellent work and open-source contributions.

### Contact
If you have any questions, please contact us via email lywei25@stu.pku.edu.cn.

