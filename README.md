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


