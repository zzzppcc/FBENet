# FBENet
### Data Preparation
Download the following datasets and unzip them into `data` folder
- [DUTS](http://saliencydetection.net/duts/)
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [PASCAL-S](http://cbi.gatech.edu/salobj/)
 
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
### Test
#### Pretrained Models
Download them from the following urls
*[[Google Drive]](https://drive.google.com/drive/folders/1Un6trEOTIVza2wH5Q2PAQVNGgsKEEHv4?usp=sharing), [[Baidu Pan,eae8]](https://pan.baidu.com/s/1xJNJ8SEDwKMHxlFh3yCUeQ?pwd=eae8)
#### Generate Saliency Maps
```
python run_test.py
```
#### Pre-computed Saliency maps
*[[Google Drive]](https://drive.google.com/drive/folders/1Un6trEOTIVza2wH5Q2PAQVNGgsKEEHv4?usp=sharing), [[Baidu Pan,eae8]](https://pan.baidu.com/s/1xJNJ8SEDwKMHxlFh3yCUeQ?pwd=eae8)

