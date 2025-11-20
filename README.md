
<p align="center">
  <!-- <img src="docs/figs/logo.png" align="center" width="50%"> -->
  
  <h3 align="center"><strong>[TVCG 2025] Zero-Shot Video Translation via Token Warping</strong></h3>

<div align="center">

<a href='http://arxiv.org/abs/2507.09168'><img src='https://img.shields.io/badge/arXiv-2311.14521-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

Training-free and inversion-free, zero-shot video translation by Token Warping in self-attention
<img src='https://github.com/Alex-Zhu1/TokenWarping/blob/main/assert/teaser.png'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</div>

## Contents
<!-- - [Demo Videos](#demo-videos)
- [Release](#release) -->
- [Contents](#contents)
- [Installation](#installation)
- [Command Line](#command-line)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

Our environment was tested on Ubuntu 22, CUDA 11.7 with 3090.

We provide an [environment.yaml](https://github.com/Alex-Zhu1/TokenWarping/blob/main/environment.yaml) file to help you verify.

## Command Line

Please try our demo by running [main_hack.py](https://github.com/Alex-Zhu1/TokenWarping/blob/main/main_hack.py).

## Citation

If you find our work helpful in your project, please cite:

```BiBTeX
@misc{zhu2025stablescoredistillation,
      title={Stable Score Distillation}, 
      author={Haiming Zhu and Yangyang Xu and Chenshu Xu and Tingrui Shen and Wenxi Liu and Yong Du and Jun Yu and Shengfeng He},
      year={2025},
      eprint={2507.09168},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.09168}, 
}
```

## Acknowledgement

Most of our code is adapted from the excellent works of [RFESCO](https://github.com/williamyang1991/FRESCO) and [ControlVideo](https://github.com/YBYBZhang/ControlVideo). We sincerely thank the authors for their great contributions.