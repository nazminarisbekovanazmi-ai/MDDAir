# MDDAir: Multi-Degradation Detection Model for All-in-One Image Restoration

Narisbekova Nazmi Narisbekovna, Juheon Park, Minyoung Jeon, Jaehyup Lee

## Abstract

All-in-one image restoration aims to recover images corrupted by diverse degradations within a single unified framework. Despite recent advances, most existing methods overlook an important step: explicitly identifying what degradation an image suffers from and how severe it is before attempting restoration. Treating all degradations and all severities of a given degradation uniformly can lead to underor over-restoration and limits generalization to real-world,mixed-degradation scenarios. We address this gap with Multi-Degradation Detection Model for All-in-One Image Restoration (MDDAir), a framework built around Degradation Estimator that infers both the degradation type and its spatial severity directly from the input image, without requiring ground-truth labels during training. This degradationaware understanding is then used to guide restoration in two complementary ways: a Severity Spatial Attention module that focuses the network on the most heavily degraded regions, and a Feature-wise Linear Modulation mechanism that adapts restoration globally based on the overall degradation intensity. Together, these components allow the network to adapt both where and how much it restores, rather than applying a one-size-fits-all correction. Extensive experiments on three- and five-degradation benchmarks show that MDDAir achieves competitive or state-of-the-art performance compared to recent methods, while providing interpretable, label free degradation predictions.

---

## Model Architecture

![Model Architecture](https://github.com/nazminarisbekovanazmi-ai/MDDAir/blob/main/figures/Fig%201.png?raw=true)


## MDDAir's novelty 

![MDDAir's novelty](https://github.com/nazminarisbekovanazmi-ai/MDDAir/blob/main/figures/Fig%202.png)
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

The pretrained weights for the 3-degradation model can be downloaded here: https://drive.google.com/file/d/1YGYYmbq1wHTthVcsimac1_6FUiwqrVJu/view?usp=drive_link. 
The pretrained weights for the 5-degradation model can be downloaded  here: https://drive.google.com/file/d/1-OYxcVjzjg9kCiUwQYz644IO4hbs6fo3/view?usp=drive_link. 
Ablation study checkpoints are here: 
no degradation estimator: https://drive.google.com/file/d/1wwp4fg_1rQpSJWLxcuHt-4L7bfA6bQSN/view?usp=drive_link 
no severity spatial attention: 
no feature-wise linear modulation: 



### Results
Performance results of the MDDAir framework trained under the all-in-one setting.
**Three Distinct Degradations**:

![3D](./fig/3D.jpg)  

**Five Distinct Degradations**:

![5D](./fig/5D.jpg) 


