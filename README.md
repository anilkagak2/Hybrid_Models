# [Hybrid Models: Efficient Edge Inference by Selective Query](https://openreview.net/forum?id=jpR98ZdIm2q)

Edge devices provide inference on predictive tasks to many end-users. However, deploying deep neural networks that achieve state-of-the-art accuracy on these devices is infeasible due to edge resource constraints. Nevertheless, cloud-only processing, the de-facto standard, is also problematic, since uploading large amounts of data imposes severe communication bottlenecks. We propose a novel end-to-end hybrid learning framework that allows the edge to selectively query only those hard examples that the cloud can classify correctly. Our framework optimizes over neural architectures and trains edge predictors and routing models so that the overall accuracy remains high while minimizing the overall latency. Training a hybrid learner is difficult since we lack annotations of hard edge-examples. We introduce a novel proxy supervision in this context and show that our method adapts seamlessly and near optimally across different latency regimes. On the ImageNet dataset, our proposed method deployed on a micro-controller unit exhibits 25\% reduction in latency compared to cloud-only processing while suffering no excess loss.

## Brief Description  

![](<Hybrid-Models-Poster.png>)

[Watch the video here](https://youtu.be/44oZzYDvFi8)


## Installation

Our codebase is written using [PyTorch](https://pytorch.org). You can set up the environment using [Conda](https://www.anaconda.com/products/individual) and executing the following commands.  

```
conda create --name pytorch-1.10 python=3.9
conda activate pytorch-1.10
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Please update the last command as per your system specifications (see [PyTorch-Install](https://pytorch.org/get-started/locally/)). Although we have not explicitly tested all the recent PyTorch versions, but you should be able to run this code on PyTorch>=1.7 and Python>=3.7


Please install the following packages used in our codebase.

```
pip install tqdm
pip install thop
pip install timm==0.5.4
pip install pyyaml
```

## Training Scripts 



```
bash runner.sh
```

## Reference (Bibtex entry)


```
@inproceedings{kag2023efficient,
  title     = {Efficient Edge Inference by Selective Query},
  author    = {Anil Kag and Igor Fedorov and Aditya Gangrade and Paul Whatmough and Venkatesh Saligrama},
  booktitle = {International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=jpR98ZdIm2q}
}
```
