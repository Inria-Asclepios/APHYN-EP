# APHYN-EP : Deep Learning for Model Correction in CardiacElectrophysiological Imaging

Test code associated with [article accepted for the conference MIDL 2022](https://openreview.net/pdf?id=7MW9oh7MDKp) by Victoriya Kashtanova, Ibrahim Ayed, Andony Arrieula, Mark Potse, Patrick Gallinari and Maxime Sermesant.

<!-- <img src="images/Model_scheme.svg" width="800" title="Model structures used in this repository"> -->

## Getting Started
------
### Prerequisites
- Linux or macOS
- Python 3.7+
- CPU or NVIDIA GPU + CUDA CuDNN

### Installation
- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

### Requirements
To run the code within this repository requires [Python 3.7+](https://www.python.org/) with the following dependencies

- [`torch`](https://pytorch.org/get-started/locally/)
- [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq)
- and some standard python libraries [`matplotlib`](https://matplotlib.org/stable/users/installing.html),  [`numpy`](https://numpy.org/), [`scipy`](https://scipy.org/) etc.

which can be installed via
```
$ pip install -r requirements.txt
```

### APHYN-EP train
Try :
```bash
python train.py --name aphynep --dataroot ./data_ttp/ --batch_size 4 --estim_param_names d,t_in
```
