
<h1 align="center">DKDR: Dynamic Knowledge Distillation for Reliability in Federated Learning</h1>

<p align="center"><em><a href="https://openreview.net/pdf?id=To2fQWv0zP" target="_blank" rel="noopener noreferrer">DKDR: Dynamic Knowledge Distillation for Reliability in Federated Learning </a></p> 

<p align="center"><em>Yueyang Yuan&dagger;, Wenke Huang&dagger;, Guancheng Wan, Kaiqi Guan, He Li, Mang Ye*</em></p>

<h2>🙌 Abstract </h2>
Federated Learning (FL) has demonstrated a promising future in privacy-friendly collaboration but it faces the data heterogeneity problem. Knowledge Distillation (KD) can serve as an effective method to address this issue. However, challenges arise from the unreliability of existing distillation methods in multi-domain scenarios.
Prevalent distillation solutions primarily aim to fit the distributions of the global model directly by minimizing forward Kullback-Leibler divergence (KLD). This results in significant bias when the outputs of the global model are multi-peaked, which indicates the unreliability of distillation pathway. Meanwhile, cross-domain update conflicts can notably reduce the accuracy of the global model (teacher model) in certain domains, reflecting the unreliability of the teacher model in these domains.
In this work, we propose DKDR (Dynamic Knowledge Distillation for Reliability in Federated Learning), which dynamically assigns weights to forward and reverse KLD based on knowledge discrepancies. This enables clients to fit the outputs from the teacher precisely. Moreover, we use knowledge decoupling to identify domain experts, thus clients can acquire reliable domain knowledge from experts. Empirical results from single-domain and multi-domain image classification tasks demonstrate the effectiveness of the proposed method and the efficiency of its key modules.

<h2>📖 Overview </h2>
<p align="center">
<img center src="frame_nips2_00.png" width = "1000" alt="Architecture of DKDR.">
</p>

<h2 id="citation"> 🥳 Citation </h2>

If our research or code assists your work, kindly cite our paper as follows::

```bibtex
@inproceedings{DKDR_NeurIPS25,
  title={Dynamic Knowledge Distillation for Reliability in Federated Learning},
  author={Yuan, Yueyang and Huang, Wenke and Wan, Guancheng and Guan, Kaiqi and Li, He and Ye, Mang},
  booktitle={NeurIPS},
  year={2025}
}
```
