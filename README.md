![CAR-UNet](1.png?raw=true "CAR-UNet")
# [Channel Attention Residual U-Net for Retinal Vessel Segmentation](https://arxiv.org/abs/2004.03702)
This code is for the paper: Channel Attention Residual U-Net for Retinal Vessel Segmentation. We report state-of-the-art performances on DRIVE, CHASE DB1 and STARE datasets.

Code written by Changlu Guo, Budapest University of Technology and Economics(BME).


We train and evaluate on Ubuntu 16.04, it will also work for Windows and OS.



# MECA
In this paper we propose a Modified Efficient Channel Attention (MECA), and its specific structure is shown in the figure below. The code can be found at [attention_module.py](attention_module.py) <br>
![MECA](2.png?raw=true "MECA")
## Quick start 

Train:
Run train_drive.py or train_chase.py <br>
or tarin_stare.py(4-fold cross-validation) 

Test:
Run eval_drive.py or eval_chase.py

## Results

![Results](3.png?raw=true "Results")

## Environments
Keras 2.3.1  <br>
Tensorflow==1.14.0 <br>


## If you are inspired by our work, please cite this paper.

@misc{guo2020channel,<br>
      title={Channel Attention Residual U-Net for Retinal Vessel Segmentation}, <br>
      author={Changlu Guo and Márton Szemenyei and Yangtao Hu and Wenle Wang and Wei Zhou and Yugen Yi},<br>
      year={2020},<br>
      eprint={2004.03702},<br>
      archivePrefix={arXiv},<br>
      primaryClass={eess.IV}<br>
}
