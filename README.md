# Robust Shape Fitting for 3D Scene Abstraction

This repository contains the source code for the cuboid-based scene decomposition method described in our paper *[Robust Shape Fitting for 3D Scene Abstraction](https://arxiv.org/abs/2403.10452)*.
It is extension of our previous work *[Cuboids Revisited: Learning Robust 3D Shape Fitting to Single RGB Images](http://arxiv.org/abs/2105.02047)*.
Please refer to the following repository if you are looking for the source code of the previous version: [cuboids_revisited_cvpr21](https://github.com/fkluger/cuboids_revisited_cvpr21).

If you use this code, please cite both papers:
```
@article{kluger2024robust,
  title={Robust Shape Fitting for 3D Scene Abstraction},
  author={Kluger, Florian and Brachmann, Eric and Yang, Michael Ying and Rosenhahn, Bodo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```
```
@inproceedings{kluger2021cuboids,
  title={Cuboids Revisited: Learning Robust 3D Shape Fitting to Single RGB Images},
  author={Kluger, Florian and Ackermann, Hanno and Brachmann, Eric and Yang, Michael Ying and Rosenhahn, Bodo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

For depth estimation, we utilise [BTS](https://github.com/cogaplex-bts/bts). If you do as well, please also cite their paper:
```
@article{lee2019big,
  title={From big to small: Multi-scale local planar guidance for monocular depth estimation},
  author={Lee, Jin Han and Han, Myung-Kyu and Ko, Dong Wook and Suh, Il Hong},
  journal={arXiv preprint arXiv:1907.10326},
  year={2019}
}
```


## Installation
Get the code:
```
git clone --recurse-submodules https://github.com/fkluger/cuboids_revisited.git
cd cuboids_revisited
git submodule update --init --recursive
```

Set up the Python environment using [Anaconda](https://www.anaconda.com/): 
```
conda env create -f environment.yml
source activate cuboids_pami
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
```

## Data
### NYU Depth v2
In order to use the NYU Depth v2 dataset, you need to obtain the original 
[MAT-file](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) and convert it to a 
*version 7* MAT-file in MATLAB so that we can load it via scipy:
```
load('nyu_depth_v2_labeled.mat')
save('nyu_depth_v2_labeled.v7.mat','-v7')
```

Then, extract all images and depth maps to separate Pickle files using our helper script:
```
python util/extract_nyu_to_files.py --source nyu_depth_v2_labeled.v7.mat --destination ./datasets/nyu_depth/files
```

### Synthetic Metropolis Homographies (SMH)

Download the SMH dataset with depth maps from here: https://github.com/fkluger/smh

## Pre-trained models
Download our pre-trained models, which we used for the experiments in our paper from 
[here](https://cloud.tnt.uni-hannover.de/index.php/s/94QCTEeiZqARHpc) and place the files in the `models` directory.

If you want to train our method for RGB input, please also obtain the pre-trained weights for the BTS depth estimator 
from [here](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_densenet121.zip) and place them in
the `models` folder as well.

## Evaluation
In order to repeat the main experiments from our paper using pre-trained neural networks, you can run the following commands:

### NYU
#### Ground truth depth, numerical solver
```
python evaluate.py --load models/nyu_gt_numerical/run1/consac_weights_best.net
```
This will run our method for depth input on the NYU test set with the parameters used in the paper and report all 
evaluation metrics at the end.
Replace `run1`with `run2 ... run5` to evaluate one of the other training runs.  

#### Ground truth depth, neural solver
```
python evaluate.py --data_path datasets/nyu_depth/files --load models/nyu_gt_neural/run1/consac_weights_best.net --minsolver transformer --load_solver models/nyu_gt_neural/run1/primitive_fit_weights_best.net
```

#### RGB image, numerical solver
```
python evaluate.py --data_path datasets/nyu_depth/files --depth_model bts --load models/nyu_bts_numerical/run1/consac_weights_best.net --load_depth models/nyu_bts_numerical/run1/depth_weights_best.net 
```

#### RGB image, neural solver
```
python evaluate.py --data_path datasets/nyu_depth/files --depth_model bts --load models/nyu_bts_neural/run1/consac_weights_best.net --load_depth models/nyu_bts_neural/run1/depth_weights_best.net --minsolver transformer --load_solver models/nyu_bts_neural/run1/primitive_fit_weights_best.net
```

### Synthetic Metropolis Homographies

#### Ground truth depth, numerical solver
```
python evaluate.py --dataset smh --data_path /path/to/smh -t 0.04 --a_min 2 --a_max 30 --load models/smh_gt_numerical/run1/consac_weights_best.net --instances 16 --fitting_lr 0.5
```

#### Ground truth depth, neural solver
```
python evaluate.py --dataset smh --data_path /path/to/smh -t 0.04 --a_min 2 --a_max 30 --load models/smh_gt_neural/run1/consac_weights_best.net --minsolver transformer --load_solver models/smh_gt_neural/run1/primitive_fit_weights_best.net
```

### Additional options
#### Visualisation

Add the option `--visualise` to save plots visualising the results for each image.
Set the destination folder with the `--eval_results PATH` option.

## Training

### Pre-Training: NYU
#### Sample weight network:
```
python train.py --train_consac --hyps 32 --data_path datasets/nyu_depth/files
```
#### Neural solver:
```
python pretrain_solver.py --dataset nyu 
```

### Pre-Training: SMH
#### Sample weight network:
```
python train.py --dataset smh --data_path /path/to/smh --train_consac --hyps 32 --consac_lr 1e-6 --maximise_second_entropy 0.1 --epochs 100 --a_min 2.0 --a_min 30 --fitting_lr 0.5 
```
#### Neural solver:
```
python pretrain_solver.py --dataset smh
```

### Fine-Tuning: NYU
#### Depth input, neural solver:
```
python train.py --minsolver transformer --train_consac --train_solver --consac_lr 1e-7 --solver_lr 1e-7 --softmax_alpha 1000 --max_prob_loss 0 --minimise_corr 0 --maximise_second_entropy 0 --hyps 32 --load models/nyu_gt_numerical/run1/consac_weights_best.net --load_solver models/solver/nyu/primitive_fit_weights_best.net --data_path datasets/nyu_depth/files
```
#### RGB input, numerical solver:
```
python train.py --depth_model bts --train_consac --train_depth --consac_lr 1e-7 --depth_lr 1e-7 --softmax_alpha 1000 --max_prob_loss 0 --minimise_corr 0 --maximise_second_entropy 0 --hyps 32 --load models/nyu_gt_numerical/run1/consac_weights_best.net
```
#### RGB input, neural solver:
```
python train.py --minsolver transformer --depth_model bts --train_consac --train_solver --train_depth --consac_lr 1e-7 --solver_lr 1e-7 --depth_lr 1e-7 --softmax_alpha 1000 --max_prob_loss 0 --minimise_corr 0 --maximise_second_entropy 0 --hyps 32 --load models/nyu_gt_numerical/run1/consac_weights_best.net --load_solver models/solver/nyu/primitive_fit_weights_best.net --data_path datasets/nyu_depth/files
```

### Fine-Tuning: SMH
#### Depth input, neural solver:
```
python train.py --dataset smh --data_path /path/to/smh -t 0.04 --a_min 2 --a_max 30 --minsolver transformer --train_consac --train_solver --consac_lr 1e-8 --solver_lr 1e-8 --softmax_alpha 1000 --max_prob_loss 0 --minimise_corr 0 --maximise_second_entropy 0 --hyps 32 --load models/smh_gt_numerical/run1/consac_weights_best.net --load_solver models/solver/smh/primitive_fit_weights_best.net
```


You may want to use a separate GPU for `--depth_gpu`, as the whole pipeline does not fit on a single GPU with 12GB memory. 