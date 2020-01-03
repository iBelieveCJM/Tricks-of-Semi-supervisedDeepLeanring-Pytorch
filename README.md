# Tricks of Semi-supervised Deep Leanring --Pytorch

The repository implements following semi-supervised deep learning methods:

- **PseudoLabel 2013**: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks (ICMLW 2013)

- **PI&Tempens**: Temporal Ensembling for Semi-Supervised Learning (ICLR 2017)

- **MeanTeacher**: Mean Teachers are better Role Models (NIPS 2017)

- **ICT**: Interpolation Consistency Training for Semi-supervised Learning (IJCAI 2019)

- **MixMatch**: A Holistic Approach to Semi-Supervised Learning (2019)

This repository was created for my blog [半监督深度学习训练和实现小Tricks](https://zhuanlan.zhihu.com/p/100252944). Therefore the hyper-parameters are set for fair comparision, rather than performance.

### The environment:

- Ubuntu 16.04 + CUDA 9.0

- Python 3.6.5:: Anaconda

- PyTorch 0.4.1 and torchvision 0.2.1

### To run the code:

The script *run.sh* includes some examples. You can try it as follow:

```shell
bash run.sh [gpu_id]
```

### Some experimental results:

I haven't run all models in this repository. Some results of this repo. are shown in *./results* directory. And the following results came from this repository and the old codes which this repo. built on.

The following table shows the error rates of the CIFAR10 experiment with 4000 labeled training samples. The parameter settings are the same with the examples in *run.sh*.

|        | iPseudoLabel2013 | ePseudoLabel2013 | iTempens | eTempens | PI    | MeanTeacher | ICT\* | MixMatch |
|------- | ---------------- | ---------------- | -------- | -------- | ------| ----------- | ----- | -------- |
|orginal |                  |                  |          | 12.16    | 13.20 | 12.31       | 7.29  | 6.24     |
| v1     | 20.03            | 12.03            | 14.52    | 10.74    | 14.11 | 10.59       | 7.12  | 6.70     |
| v2     |                  |                  |          |          |       |  9.46       | 6.43  |          |


Note:

- The MeanTeacher is the first model of the repository. So the hyper-parameters actually have been tuned for MeanTeacher.

- My ICT is different from original one. The main difference is the unsupervised loss for unlabeled data.
