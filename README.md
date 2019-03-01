# OPPO AI -- Face Recognition

## Environment:

- Ubuntu 16.04
- pytorch=0.4.1 or pytorch=0.1 (Both compatible)
- Anaconda3, python3.6  
 
## How to test: 

- Install pytorch and other requirements by `pip install requirements`. 
- `cd ROOT_PATH`. Make sure execute following cmd in the root directory of project. 
- `ln -s OPPO_DATASET ./`
- `python test.py --data_dir OPPO_DATASET`. We contain a image folder `images_aligned_sample` for demonstration, thus `python test.py --data_dir images_aligned_sample` should run successfully. 

We are not sure whether the program is bug-free. If there is any mistake, please inform us(xingluwang@zju.edu.cn). Thank you!
