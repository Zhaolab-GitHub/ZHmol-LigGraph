# [ZHmol-LigGraph](http://zhaoserver.com.cn/ZHmol-LigGraph/ZHmol-LigGraph.html)

## Overview of ZHmol-LigGraph: 
ZHmol-LigGraph is a graph-based deep learning method for nucleic acid-ligand complex prediction, featuring a two-step architecture that jointly models ligand flexibility and nucleic acid surface interactions at atomic resolution.

**We also provide an online calculation server for convenient use:** 
[Click to access the online server](http://zhaoserver.com.cn/ZHmol-LigGraph/ZHmol-LigGraph.html)

## Environment Setup

### Prerequisites:
* OS: Ubuntu 20.04 LTS (recommended)
* NVIDIA GeForce 2080
* Python 3.8.18
* CUDA 10.2

### Installation Steps:
Follow these steps to set up your environment. We have successfully tested this in a clean environment:  
(1) Create a new conda environment:
```
conda create -n ZHmol-LigGraph python=3.8.18
conda activate ZHmol-LigGraph
```

(2) Install necessary dependencies:
```
pip install h5py==3.10.0
pip install scipy==1.10.1
pip install tqdm==4.66.1
pip install pandas==2.0.3
pip install torch==1.7.1
```

(3) Install PyG extensions (download required)
```
pip install torch_cluster-1.5.8-cp38-cp38-linux_x86_64.whl (need download)
pip install torch_scatter-2.0.5-cp38-cp38-linux_x86_64.whl (need download)
pip install torch_sparse-0.6.8-cp38-cp38-linux_x86_64.whl (need download)
pip install torch_geometric==1.6.3
pip install torchvision==0.8.2
```

### A full list of dependencies can be found in: requirements.txt

## Data & Pretrained Models
* Data Files​​: Available at at XX
* Pretrained Models​​: Download prediction_model.pt and selection_model.pt from [LINK] and place them in models/ directory.

## Usage

### Case 1: With Initial Conformation
If you already have the initial nucleic acid-ligand complex structure, run:

```
bash run_ZHmol-LigGraph.sh # Single or multiple inputs
```

### Case 2: Without Initial Conformation
#### If you do not have the initial conformation, we recommend generating it using XDock:
(1) Download and install [XDock](http://huanglab.phys.hust.edu.cn/XDock/).  
(2) Then run the bash script:
```
bash run_ZHmol-LigGraph.sh # Single or multiple inputs
```

### Output
Results are saved in data/after_dock/ as:
* .mol2 files (predicted 3D structures)
* .txt file (scores of each predicted pose)

## Contact
If you have any comments, questions or suggestions about the ZHmol-LigGraph, please contact:  
Yunjie Zhao; E-mail: yjzhaowh@ccnu.edu.cn  
Chengwei Zeng; E-mail: cwzengwuhan@mails.ccnu.edu.cn
