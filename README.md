# RACOD-Net
## :fire: Camouflage Object Detection & Segmentation  :fire: 
# Table of contents
- [RACOD-Net](#racod-net)
- [Table of contents](#table-of-contents)
  - [RACOD-Net ](#racod-net-)
  - [General Instructions ](#general-instructions-)
  - [Datasets ](#datasets-)
  - [Produced Weights  ](#produced-weights--)
  - [Results ](#Results-)
  - [Qualitative Comparison ](#qualitative-comparison-)
  - [License ](#licence-)
  
## RACOD-Net <a name="introduction"></a>
Unlike previous studies RACOD architecture manages to successfully combine two powerful backbone encoders, a CNN encoder and a Transformer encoder, through a novel partial cascaded decoder to output an enriched tensor containing both global and local information.
As shown in the figure below, displaying the complete architechure, we initially have a total of 7 features from our backbone encoders. A set of 3 features extracted from ResNet50 encoder and a set of 4 features extracted from SegFormer encoder.
![Test Image 4](https://github.com/mikestratakis/RACOD-Net/blob/master/ShowCase-RACOD-Net/completearch.png)

## General Instructions <a name="General Instructions"></a>
In order to execute RACOD-Net you need to install certain libraries. We advice following the next steps, starting from a clean setup of your system.
- https://developer.nvidia.com/cuda-downloads
- Install nvidia drivers for your system from: https://developer.nvidia.com/cuda-downloads
- Install miniconda3 to run your future code in a virtual environment
- Based from the cuda version downloaded from step 1, install cuda from: https://pytorch.org/get-started/locally/ 
- Before executing any code choose the python interpeter from the miniconda environment
- Afterwards install the following packages:
    - Install timm with conda: conda install -c conda-forge timm
    - Install matplot with conda: conda install -c conda-forge matplotlib
    - Install scipy with conda: conda install -c anaconda scipy
    - Install cv2 with conda: conda install -c conda-forge opencv

## Datasets <a name="Datasets"></a>
Download the train and test datasets for Camouflaged Object Detection and Polyp Detection from the following links:
- Camouflaged Object Datasets: :boom: Link coming soon :boom:
- Polyp Segmentation Datasets: :boom: Link coming soon :boom:

Afterwards place the downloaded compressed files inside folder Datasets and decompress.

## Produced Weights  <a name="Produced Weights"></a>
For camouflaged ovject detection we provide our best weights, setting new records over many evaluation metrics, from the following link:
- :boom: Link coming soon :boom:

After successfully downloading the weights place them inside the folder Produced_Weights/RACOD/COD/ <br/>
For polyp segmentation we added the files Train_Polyp and Evaluation_Polyp to train and evaluate over polyp datasets. <br/>
The best weights for polyp segmentation will be soon provided.


## :fire: Results :fire: <a name="Results"></a>
We argue that our final segmentation results are very close to the ground-truth annotations, by successfully segmenting not only large camouflaged objects but also small ones. From the figure below and various other results we observed that our method successfully segments the position of camouflage objects with accurate and precise boundaries over several challenging scenes, such as multiple and low-contrast objects. Even when some camouflaged objects are divided into separate parts because of the interference with other non-camouflaged objects RACOD-Net is still capable of detecting and segmenting the expected target.
Download our best visual results from the following links:
- Visual Results: :boom: Link coming soon :boom:

<p align="center">
  <img src="https://github.com/mikestratakis/RACOD-Net/blob/master/ShowCase-RACOD-Net/visual_results.png" />
</p>

## :fire: Qualitative Comparison :fire: <a name="Qualitative Comparison"></a>
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
