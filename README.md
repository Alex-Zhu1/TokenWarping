<!-- <h1 align="center"><strong>TokenWarping</strong></h1> -->
<h3 align="center">Zero-Shot Video Translation via Token Warping</h3>
<h4 align="center">IEEE TVCG 2025</h4>

<div align="center">

[![arXiv paper 2402.12099 badge with red background](https://img.shields.io/badge/arXiv-2507.09168-b31b1b.svg)](http://arxiv.org/abs/2402.12099) [![Project page link badge with blue background](https://img.shields.io/badge/Project-Page-blue)](https://alex-zhu1.github.io/TokenWarping/)

</div>

<p align="center">
  ⚡ <strong>Training-free • Inversion-free • Zero-shot</strong> ⚡<br>
  Video-to-video translation via token warping in self-attention
</p>

<p align="center">
  <a href="https://github.com/Alex-Zhu1/TokenWarping">
    <img src="assert/teaser.png" alt="TokenWarping Teaser" width="100%">
  </a>
</p>

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

Please try our demo by running [`main_hack.py`](https://github.com/Alex-Zhu1/TokenWarping/blob/main/main_hack.py)

Due to limited GPU memory, we also provide a **batch-infer** mode for longer videos.  
To enable it, modify **main_hack.py** as follows:

```python
from infer_batch import main  # flow prediction uses 8-frame batch-infer
```

With batch-infer enabled, the pipeline can process up to ~120 frames on a typical 24 GB GPU.

## Citation

If you find our work helpful in your project, please cite:

```BiBTeX
@misc{zhu2025zeroshotvideotranslationtoken,
      title={Zero-Shot Video Translation via Token Warping}, 
      author={Haiming Zhu and Yangyang Xu and Jun Yu and Shengfeng He},
      year={2025},
      eprint={2402.12099},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2402.12099}, 
}
```

## Acknowledgement

Most of our code is adapted from the excellent works of [RFESCO](https://github.com/williamyang1991/FRESCO) and [ControlVideo](https://github.com/YBYBZhang/ControlVideo). We sincerely thank the authors for their great contributions.