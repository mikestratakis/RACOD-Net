# RACOD-Net
## :fire: Camouflage Object Detection & Segmentation  :fire: 
# Table of contents
- [RACOD-Net](#racod-net)
- [Table of contents](#table-of-contents)
  - [RACOD-Net ](#racod-net-)
  - [General Instructions ](#general-instructions-)
  - [Notebook Demo](#notebook-demo-)
  - [Datasets ](#datasets-)
  - [Produced Weights ](#produced-weights--)
  - [Results ](#Results-)
  - [Qualitative Comparison ](#qualitative-comparison-)
  - [License ](#licence-)
  
## RACOD-Net <a name="introduction"></a>
Unlike previous studies RACOD-Net architecture manages to successfully combine two powerful backbone encoders, a CNN encoder and a Transformer encoder, through a novel partial cascaded decoder to output an enriched tensor containing both global and local information.
As shown in the figure below, displaying the complete architechure, we initially have a total of 7 features from our backbone encoders. A set of 3 features extracted from ResNet50 encoder and a set of 4 features extracted from SegFormer encoder.
![Test Image 4](https://github.com/mikestratakis/RACOD-Net/blob/master/ShowCase-RACOD-Net/completearch.png)

## Notebook Demo <a name="Notebook-Demo"></a>

Available Demo of RACOD-Net architecture can be found in online community platform kaggle from the following link:
- https://www.kaggle.com/code/michailstratakis/racod-net

## General Instructions <a name="General Instructions"></a>
## Update 4/4/2024
In order to execute RACOD-Net you need to install certain libraries. We advice following the next steps, starting from a clean setup of your system. For this particular deep learning achitecture we used Ubuntu 22.04 as our operating system and an Zotac GeForce RTX 3060 12GB GDDR6 Twin Edge as our Graphic Card. 
Also it is required to have at least 15gb on your installation directory to proceed successfully in the installation process.

<ins>Optional instructions</ins><br>
Generally we suggest installing a virtual environment like miniconda3 to execute Racod-Net in a virtual environment.<br>
The following instructions will install miniconda3 in your system:
  - wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  - bash Miniconda3-latest-Linux-x86_64.sh
  - conda --version

If you have installed miniconda3 restart terminal before proceeding further.

The following instructions will apply the cuda drivers on Ubuntu 22.04 and execute Racod-Net successfully.<br>
On a clean setup execute the following:
- sudo apt-get update
- sudo apt-get upgrade
- Based from https://developer.nvidia.com/cuda-downloads execute the following:
  - wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
  - sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
  - wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
  - sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
  - sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
  - sudo apt-get update
  - sudo apt-get -y install cuda-toolkit-12-4
  - sudo apt-get install -y cuda-drivers <br>
  Now you have installed the 550 proprietary metapackage nvidia driver.
  Run the following to check if cuda drivers are installed:
    - nvcc --version
      - If nvcc is not found run the following:
      - sudo apt install nvidia-cuda-toolkit
      - nvcc --version 
      - It is now expected that a Cuda compiler driver is intalled in your system

Based from https://pytorch.org/get-started/locally/ execute the following:
  - conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

Afterwards install the following packages that are required from Racod-Net:
  - Install timm with conda: conda install -c conda-forge timm
  - Install matplot with conda: conda install -c conda-forge matplotlib
  - Install scipy with conda: conda install -c anaconda scipy
  - Install cv2 with conda: conda install -c conda-forge opencv

<ins>Visual Studio Integration</ins>
  - If using visual studio make sure to select the virtual miniconda3 python interpreter before executing Racod-Net

## Datasets <a name="Datasets"></a>
Download the train and test datasets for Camouflaged Object Detection and Polyp Detection from the following links:
- Camouflaged Object Datasets: :boom: Link coming soon :boom:
- Polyp Segmentation Datasets: :boom: Link coming soon :boom:

Afterwards place the downloaded compressed files inside folder Datasets and decompress.
Also download the mit-b4 pretrained weights, required for the initialization of SegFormer encoder used by RACOD-Net, from the following link:
- mit-b4 weights: :boom: Link coming soon :boom:

Afterward place the dowloaded weights inside folder: Pretrained_Weights_SegFormer <br/>
SegFormer encoder is based by the original paper: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"

## Produced Weights  <a name="Produced Weights"></a>
For camouflaged object detection we provide our best weights, setting new records over many evaluation metrics, from the following link:
- :boom: Link coming soon :boom:

After successfully downloading the weights place them inside the folder Produced_Weights/RACOD/COD/ <br/>
For polyp segmentation we added the files Train_Polyp and Evaluation_Polyp to train and evaluate over polyp datasets. <br/>
The best weights for polyp segmentation will be soon provided.


## Results <a name="Results"></a>
We argue that our final segmentation results are very close to the ground-truth annotations, by successfully segmenting not only large camouflaged objects but also small ones. From the figure below and various other results we observed that our method successfully segments the position of camouflage objects with accurate and precise boundaries over several challenging scenes, such as multiple and low-contrast objects. Even when some camouflaged objects are divided into separate parts because of the interference with other non-camouflaged objects RACOD-Net is still capable of detecting and segmenting the expected target.
Download our best visual results from the following links:
- Visual Results: :boom: Link coming soon :boom:

<p align="center">
  <img src="https://github.com/mikestratakis/RACOD-Net/blob/master/ShowCase-RACOD-Net/visual_results.png" />
</p>

## Qualitative Comparison <a name="Qualitative Comparison"></a>
We compare RACOD-Net with several state-of-the-art camouflaged object detection studies that deploys either a CNN or a Transformer based arhitechure. As shown in Tab. 1 RACOD-Net surpasses almost all methods in several evaluation metrics in almost all datasets. Even when RACOD-Net's predictions come second they are still very close from reaching the top.
For fair comparison, all the predictions are evaluated using the same evaluation metrics and the same evaluation code. Additionally, all the camouflaged maps prediction scores are provided either by the authors or generated
by retraining the models with the provided open source codes.
<p align="center">
  <img src="https://github.com/mikestratakis/RACOD-Net/blob/master/ShowCase-RACOD-Net/quantitative_results.png" />
</p>

${\color{red}Red,\space \color{green}Green,\space \color{blue}Blue}$  indicate the best, second best and third best performance. ‘↑/↓’ denotes that the higher/lower the score, the better.

## Licence <a name="Licence"></a>
Copyright © – All rights reserved Mike Stratakis, 2023.
    Copying, storing and distribution of this work is prohibited,
    in whole or in part, for commercial purposes. Reprinting is permitted,
    storage and distribution for non-profit, educational or
    of a research nature, provided that the source of origin is indicated and that
    the present message is retained. Questions concerning the use of the work
    for profit should be addressed to the author.
    The opinions and conclusions contained in this document express
    the author.
