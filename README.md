# Face Rec

## Environment:

- Ubuntu 16.04
- pytorch=0.4.1 or pytorch=0.1 
- Anaconda3, python3.6 or python3.5
 
## How to test: 

- Install pytorch and other requirements by `pip install requirements.txt`. 
- `cd ROOT_PATH`. Make sure execute following cmd in the root directory of project. 
- `python test.py --data_dir OPPO_DATASET`. We contain a image folder `images_aligned_sample` for demonstration, thus `python test.py --data_dir images_aligned_sample` should run successfully. Relative path or absolute path are both accepted. 
