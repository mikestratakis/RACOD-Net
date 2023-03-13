# RACOD-Net
Camouflage Object Detection

# Table of contents
- [RACOD-Net](#racod-net)
- [Table of contents](#table-of-contents)
  - [RACOD-Net ](#racod-net-)
  - [General Instructions ](#general-instructions-)
  - [Datasets ](#datasets-)
  - [Produced Weights  ](#produced-weights--)
  - [Results ](#results-)
  - [Qualitative Comparison ](#qualitative-comparison-)

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
Download the train and test datasets for Camouflaged Object Detection and Polyp Detection from the following link:
Afterwards place the downloaded compressed files inside folder Datasets and decompress.
## Produced Weights  <a name="Produced Weights "></a>
We provide our best weights, setting new records over many evaluation metrics, from the following link:
After successfully downloading the weights place them inside the folder Produced_Weights/RACOD
## Results <a name="Results"></a>
We argue that our final segmentation results are very close to the ground-truth annotations, by successfully segmenting not only large camouflaged objects but also small ones. From the figure below and various other results we observed that our method successfully segments the position of camouflage objects with accurate and precise boundaries over several challenging scenes, such as multiple and low-contrast objects. Even when some camouflaged objects are divided into separate parts because of the interference with other non-camouflaged objects RACOD-Net is still capable of detecting and segmenting the expected target.
<p align="center">
  <img src="https://github.com/mikestratakis/RACOD-Net/blob/master/ShowCase-RACOD-Net/comparative_results.png" />
</p>
## Qualitative Comparison <a name="Qualitative Comparison"></a>
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. 
