# MDDAir: Multi-Degradation Detection Model for All-in-One Image Restoration

Narisbekova Nazmi Narisbekovna, Juheon Park, Minyoung Jeon, Jaehyup Lee

## Abstract

All-in-one image restoration aims to recover images corrupted by diverse degradations within a single unified framework. Despite recent advances, most existing methods overlook an important step: explicitly identifying what degradation an image suffers from and how severe it is before attempting restoration. Treating all degradations and all severities of a given degradation uniformly can lead to underor over-restoration and limits generalization to real-world,mixed-degradation scenarios. We address this gap with Multi-Degradation Detection Model for All-in-One Image Restoration (MDDAir), a framework built around Degradation Estimator that infers both the degradation type and its spatial severity directly from the input image, without requiring ground-truth labels during training. This degradationaware understanding is then used to guide restoration in two complementary ways: a Severity Spatial Attention module that focuses the network on the most heavily degraded regions, and a Feature-wise Linear Modulation mechanism that adapts restoration globally based on the overall degradation intensity. Together, these components allow the network to adapt both where and how much it restores, rather than applying a one-size-fits-all correction. Extensive experiments on three- and five-degradation benchmarks show that MDDAir achieves competitive or state-of-the-art performance compared to recent methods, while providing interpretable, label free degradation predictions.

---

## Model Architecture

![Model Architecture](./fig/shuffle-fram.jpg)  

---

## Usage

### Training

Train three types of degradations by running:

```bash
python train_3D_MDDAir.py
```
Train five types of degradations by running:
```bash
python train_5D_MDDAir.py
```
###  Testing

Test three types of degradations by running:

```bash
python test_3D_MDDAir.py
```
Test five types of degradations by running:

```bash
python test_5D_MDDAir.py
```
### Pretrained Model Weights

The pretrained weights for the three distinct degradation types under simple prompts are provided [here](https://pan.baidu.com/s/1W8mjjSB4XiL70cVK9B9Eng  )(pa3a), while detail prompts are provided [here](https://pan.baidu.com/s/1hk5JgOpl3VYEsWEecPpJkg?pwd=tjkd  )(tjkd). The pretrained weights for the five degradation types are available [here](https://pan.baidu.com/s/1LhAsRq8t4dvaD-hC6yDZrA?pwd=q0sm)(q0sm) .

We have also uploaded and shared the model weights on Google Drive. The link is: https://drive.google.com/drive/folders/15bOk4xrsK1b3nIu3-OzUj4O_ZssdFN5Y?usp=sharing

### Results
Performance results of the MDDAir framework trained under the all-in-one setting.
**Three Distinct Degradations**:

![3D](./fig/3D.jpg)  

**Five Distinct Degradations**:

![5D](./fig/5D.jpg) 

The visual results under the three degradation types are provided [here](https://pan.baidu.com/s/1xa_i7cbg5slEyLvBpC4JKg?pwd=lljd )(lljd).  The visual results under the five degradation types are provided [here](https://pan.baidu.com/s/1tfYrxfOI61om8QX9PnXLFA?pwd=tsbp)(tsbp).

### Python Runtime Environment
python: 3.11.5

pytorch: 2.1.1

numpy: 1.26.0

### Citation
If you use our work, please consider citing:
```bash
@inproceedings{tian2025degradation,
  title={Degradation-Aware Feature Perturbation for All-in-One Image Restoration},
  author={Tian, Xiangpeng and Liao, Xiangyu and Liu, Xiao and Li, Meng and Ren, Chao},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={28165--28175},
  year={2025}
}
```
## Contact

Should you have any questions, please contact tianxp@stu.scu.edu.cn

**Acknowledgment:** This code is based on the [PromptIR](https://github.com/va1shn9v/PromptIR) and [PIP]([longzilicart/pip_universal](https://github.com/longzilicart/pip_universal)) repository.
