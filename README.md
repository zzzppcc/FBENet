# FBENet
### Data Preparation
The overall file structure is as follows:
```shell
data
├── DUTS
│   ├── image
│   ├── mask
│   ├── edge
│   ├── train.txt
│   ├── test.txt
├── DUT-OMRON
│   ├── image
│   ├── mask
│   ├── test.txt
├── ECSSD
│   ├── image
│   ├── mask
│   ├── test.txt
├── HKU-IS
│   ├── image
│   ├── mask
│   ├── test.txt
├── PASCAL-S
│   ├── image
│   ├── mask
│   ├── test.txt
```
### Requirements

* python 3.7
* apex 0.1
* pytorch 1.9.0+cu111
* opencv-python 4.5.5
* numpy 1.21.5


### Test
#### Pretrained Models
Download them from the following urls

#### Generate Saliency Maps and Evaluate
```
python run_test.py
python evaluate.py
```
#### Pre-computed Saliency maps
Download them from the following urls


