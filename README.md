
This code repository contains an implementation of our ECCV video detection work. If you use this code, please cite:

[Video Object Detection with an Aligned Spatial-Temporal Memory](http://fanyix.cs.ucdavis.edu/project/stmn/project.html), [Fanyi Xiao](http://fanyix.cs.ucdavis.edu/) and [Yong Jae Lee](http://web.cs.ucdavis.edu/~yjlee/) in ECCV 2018. [\[Bibtex\]](http://fanyix.cs.ucdavis.edu/project/stmn/bib.txt)

## Acknowledgement
We develop this codebase from the great code of [multipathnet](https://github.com/facebookresearch/multipathnet). 

## Getting Started
### Installation
The following installation procedure is tested under:
```
Ubuntu 16.04
CUDA 9.0
Torch 7
```

- Create a directory that we call ```$ROOT``` (here we set ```$ROOT``` as ```~/code/VID``` for example):
```bash
mkdir ~/code/VID
```

- Go to ```$ROOT``` and clone this repo. You should see ```$ROOT/STMN``` after this command. 
```bash
cd $ROOT
git clone https://github.com/fanyix/STMN.git
```

- Download ImageNet VID dataset (ILSVRC 2015) and unzip it under ```$ROOT/dataset/ImageNetVID```. Note that you need to download both ```ILSVRC2015_VID_initial.tar.gz``` and ```ILSVRC2015_VID_final.tar.gz```. As a sanity check, you should be able to see ```train```, ```val``` and ```test``` under ```$ROOT/dataset/ImageNetVID/Data/VID/```. 

- Create ```$ROOT/dataset/ImageNetVID/exp```. Download the [annotations](https://drive.google.com/open?id=1GiNRfhrm8lvD66AAnLf4XEfjWHpTkW1M) and the [proposals](https://drive.google.com/open?id=1hyQPeJZCkn9Dpt3SGNGXOmlux0M-EXHV) into this directory. 

- Create ```$ROOT/dataset/ImageNetVID/models```. Download pre-trained models into this directory. Specifically, here are some models you might want to use: 1) The first one is our pre-trained [STMN model](https://drive.google.com/open?id=1rUchPUSRrksDp4JRboj2h1z-GfG1zbDu). 2) The second one is the pre-trained [RFCN model](https://drive.google.com/open?id=14Cb39nOP9DA0dl6GxY2D_KzWLVnTTmbL). 3) You also need the ImageNet classification pre-trained [ResNet-101 model](https://drive.google.com/open?id=1za16fuJACTOXsg4e1-h54QpLRYmh1PgM). 

- Download ImageNet DET dataset into ```$ROOT/dataset/ImageNetDET```. Then download from here the [proposal and annotation files](https://drive.google.com/open?id=18HE16V_N2epuLrZ0j-yklyaO4mRURGzR) we prepared, and unzip it into the ImageNet DET directory. 

After these steps, you should be expecting a code/data structure like the following:

```
$ROOT
  - STMN
  - dataset
    - ImageNetVID
      - Data
        - VID
          - train
          - val
          - test
      - exp
        - anno
          - train.t7
          - val.t7
          - test.t7
        - proposals
          - train
          - val
          - test
      - models
        - stmn.t7
        - rfcn.t7
        - resnet-101.t7
    - ImageNetDET
      - Annotations
      - ImageSets
      - Data
        - DET
          - train
          - val
      - exp
        - annotations
        - proposals
```

- Install Lua Torch following the instructions on 
```
http://torch.ch/docs/getting-started.html#_
```
Note if you are using CUDA 9.0, you probably need [this](https://github.com/torch/cutorch/issues/797) to solve a compilation issue regarding half precision operation. 

- Install necessary libraries for Torch
```
luarocks install torchnet
luarocks install optnet
luarocks install inn
luarocks install class
```

- Install hdf5 for Torch
```
sudo apt-get install libhdf5-serial-dev hdf5-tools
git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
luarocks make hdf5-0-0.rockspec LIBHDF5_LIBDIR="/usr/lib/x86_64-linux-gnu/"
```

- Install matio for Torch
```
sudo apt-get install libmatio2
luarocks install matio
```

- Install utilities for R-FCN, MatchTrans and STN modules

```
cd $ROOT/STMN/modules/rfcn
luarocks make rfcn-0.1-0.rockspec

cd $ROOT/STMN/modules/assemble
luarocks make assemble-0.1-0.rockspec

cd $ROOT/STMN/modules/stnbhwd
luarocks make stnbhwd-scm-1.rockspec
```

- Install Lua API for COCO dataset

```
cd $ROOT/STMN/external/coco
luarocks make LuaAPI/rocks/coco-scm-1.rockspec
```

- [**Optional**] Install the temporal linkage code modified from [D&T](https://github.com/feichtenhofer/Detect-Track) (MATLAB required). Note you will need this if you want to reproduce our results on ImageNet VID. Go to ```$ROOT/STMN/external/dp``` and run ```rfcn_build.m``` in MATLAB.


- [**Optional**] Finally, it seems CUDNN 7.0 is not working well with CUDA 9.0 for Torch, the following GitHub issue solves this problem: https://github.com/soumith/cudnn.torch/issues/383



### Training models

- To train an RFCN model, go to ```$ROOT/STMN``` and run the following (we use 2 V100 GPUs for training):
```
CUDA_VISIBLE_DEVICES=0,1 th train_video.lua -model rfcn -ckpt rfcn
```

- To train an STMN model, go to ```$ROOT/STMN``` and run the following:
```
CUDA_VISIBLE_DEVICES=0,1 th train_video.lua -model stmn -ckpt stmn
```

### Evaluating models

- To generate detections with the pre-trained RFCN model, go to ```$ROOT/STMN/scripts``` and run the following:
```
CUDA_VISIBLE_DEVICES=0 th eval_detect_full.lua -model rfcn -model_path ../../dataset/ImageNetVID/models/rfcn.t7 -ckpt rfcn_eval
```

- To generate detections with the pre-trained STMN model, go to ```$ROOT/STMN/scripts``` and run the following:
```
CUDA_VISIBLE_DEVICES=0 th eval_detect_full.lua -model stmn -model_path ../../dataset/ImageNetVID/models/stmn.t7 -ckpt stmn_eval
```

Please note that above commands are examples following which you can reproduce our results, however it will be slow due to the sheer amount of frames you need to evaluate. Instead in our own experiments we always parallelize the above procedure with the help of the launch script provided in ```scripts/launcher.py```. We highly encourage you to take a look at this script and parallelize this procedure like we do. 

Okay, after you're done with both commands shown above, you should have produced the raw detection results (without NMS) which we will then send to the temporal linkage procedure to generate our final detections. For this, we base our code on the brilliant code of [D&T](https://github.com/feichtenhofer/Detect-Track) (however it does require a MATLAB license to use this code) and make some modifications to <em>only use its dynamic programming functionality</em>. 

- To generate final detections with temporal linkage, go to ```$ROOT/STMN/external/dp```, run ```run_dp.m``` in MATLAB.  

[**Optional**] Again, you will reproduce our results (80.5% mAP) with the above command, however it might be slow to go over the evaluation set. To assist you in this process, we also provide some parallelization utilities in ```run_dp.m``` and a launch script ```launcher.py``` under ```$ROOT/STMN/external/dp``` (note this is a different launch script than the one we used above under ```$ROOT/STMN/scripts/```). Specifically, you first set the ```opts.scan_det``` in ```run_dp.m``` to ```true``` and launch it with ```$ROOT/STMN/external/dp/launcher.py```. Then you set ```opts.scan_det``` in ```run_dp.m``` to ```false``` and ```opts.load_scan_det``` to ```true``` and run the script again in a MATLAB console. 










