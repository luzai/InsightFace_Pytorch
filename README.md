# OPPO AI -- Face Recognition -- Version 3 

## Environment:

- Ubuntu 16.04
- pytorch=0.4.1 or pytorch=0.1 
- Anaconda3, python3.6 or python3.5
 
## How to test: 

- Install pytorch and other requirements by `pip install requirements.txt`. 
- `cd ROOT_PATH`. Make sure execute following cmd in the root directory of project. 
- `python test.py --data_dir OPPO_DATASET`. We contain a image folder `images_aligned_sample` for demonstration, thus `python test.py --data_dir images_aligned_sample` should run successfully. Relative path or absolute path are both accepted. 

We are not sure whether the program is bug-free. If there is any mistake, please inform us(xingluwang@zju.edu.cn). Thank you!

## Version 2 update 
- compatible with python3.6
- Make sure future-fstrings is installed 

## Version 3 update 
- Accelerate feature extraction part for test stage

## Version 4 update 
- Accelerate face cropping part 
- dlib==19.10 may take some time to be installed, please be patient 
- If the input image resolution is large, use '--batch_size' to reduce batch size.