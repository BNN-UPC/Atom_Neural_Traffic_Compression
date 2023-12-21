# Atom: Neural Traffic Compression with Spatio-Temporal Graph Neural Networks
#### Link to paper: [[here](https://arxiv.org/abs/2311.05337)]
#### P. Almasan, K. Rusek, S. Xiao, X. Shi, X. Cheng, A. Cabellos-Aparicio, P. Barlet-Ros
 
Contact: <paulalmasan@gmail.com>

[![Twitter Follow](https://img.shields.io/twitter/follow/PaulAlmasan?style=social)](https://twitter.com/PaulAlmasan)
[![GitHub watchers](https://img.shields.io/github/watchers/BNN-UPC/Atom_Neural_Traffic_Compression?style=social&label=Watch)](https://github.com/BNN-UPC/Atom_Neural_Traffic_Compression)
[![GitHub forks](https://img.shields.io/github/forks/BNN-UPC/Atom_Neural_Traffic_Compression?style=social&label=Fork)](https://github.com/BNN-UPC/Atom_Neural_Traffic_Compression)
[![GitHub stars](https://img.shields.io/github/stars/BNN-UPC/Atom_Neural_Traffic_Compression?style=social&label=Star)](https://github.com/BNN-UPC/Atom_Neural_Traffic_Compression)

## Abstract
Storing network traffic data is key to efficient network management; however, it is becoming more challenging and costly due to the ever-increasing data transmission rates, traffic volumes, and connected devices. In this paper, we explore the use of neural architectures for network traffic compression. Specifically, we consider a network scenario with multiple measurement points in a network topology. Such measurements can be interpreted as multiple time series that exhibit spatial and temporal correlations induced by network topology, routing, or user behavior. We present Atom, a neural traffic compression method that leverages spatial and temporal correlations present in network traffic. Atom implements a customized spatio-temporal graph neural network design that effectively exploits both types of correlations simultaneously. The experimental results show that Atom can outperform GZIP's compression ratios by 50%--65% on three real-world networks.  

# Instructions to execute

[See the execution instructions](https://github.com/BNN-UPC/Atom_Neural_Traffic_Compression/tree/main/code#instructions-to-execute)

## Description

To know more details about the implementation used in the experiments contact: [paulalmasan@gmail.com](mailto:paulalmasan@gmail.com)

Please cite the corresponding article if you use the code from this repository:

```
@inproceedings{10.1145/3630049.3630170,
author = {Almasan, Paul and Rusek, Krzysztof and Xiao, Shihan and Shi, Xiang and Cheng, Xiangle and Cabellos-Aparicio, Albert and Barlet-Ros, Pere},
title = {Atom: Neural Traffic Compression with Spatio-Temporal Graph Neural Networks},
year = {2023},
isbn = {9798400704482},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3630049.3630170},
doi = {10.1145/3630049.3630170},
abstract = {Storing network traffic data is key to efficient network management; however, it is becoming more challenging and costly due to the ever-increasing data transmission rates, traffic volumes, and connected devices. In this paper, we explore the use of neural architectures for network traffic compression. Specifically, we consider a network scenario with multiple measurement points in a network topology. Such measurements can be interpreted as multiple time series that exhibit spatial and temporal correlations induced by network topology, routing, or user behavior. We present Atom, a neural traffic compression method that leverages spatial and temporal correlations present in network traffic. Atom implements a customized spatio-temporal graph neural network design that effectively exploits both types of correlations simultaneously. The experimental results show that Atom can outperform GZIP's compression ratios by 50\%--65\% on three real-world networks.},
booktitle = {Proceedings of the 2nd on Graph Neural Networking Workshop 2023},
pages = {1–6},
numpages = {6},
keywords = {neural traffic compression, spatio-temporal graph neural networks},
location = {<conf-loc>, <city>Paris</city>, <country>France</country>, </conf-loc>},
series = {GNNet '23}
}
```