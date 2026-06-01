# DKDR: Dynamic Knowledge Distillation for Reliability in Federated Learning

[Paper link](https://openreview.net/pdf?id=To2fQWv0zP)

This repository contains a simplified public release of DKDR that matches the method structure and experiment scope described in the NeurIPS 2025 paper.

## Supported scope

- Methods: `DKDR`, `FedAVG`
- Settings: `label_skew`, `domain_skew`
- Single-domain datasets: `fl_cifar10`, `fl_cifar100`
- Multi-domain datasets: `Office31`, `OfficeHome`
- Backbones: `fedavg_cifar`, `resnet10`, `resnet18`

The public repository intentionally removes unrelated attack pipelines, OOD branches, extra dataset variants, and cached artifacts so that the code path stays aligned with the paper.

## Example usage

```bash
python main.py --task label_skew --dataset fl_cifar10 --method DKDR
python main.py --task domain_skew --dataset Office31 --method DKDR
```

## Overview

![Architecture of DKDR.](frame_nips2_00.png)

## Citation

```bibtex
@inproceedings{DKDR_NeurIPS25,
  title={Dynamic Knowledge Distillation for Reliability in Federated Learning},
  author={Yuan, Yueyang and Huang, Wenke and Wan, Guancheng and Guan, Kaiqi and Li, He and Ye, Mang},
  booktitle={NeurIPS},
  year={2025}
}
```
