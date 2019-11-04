print('------------------------')
import os,time,shutil,sys
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# os.system('pwd&ifconfig&ls&hostname&pip list|grep torch')
print(sys.argv) 

use2243='local'
for t in sys.argv:
    if '2243' in t:
        use2243 = '2243'
    elif '6944' in t:
        use2243 = '6944'

if use2243 == '2243':
    cloud_prefix="s3://bucket-2243/zl"
elif use2243=='6944':
    cloud_prefix="s3://bucket-6944/zl"
else:
    cloud_prefix="/home/zl/prj" 

if use2243!='local':
    import moxing, moxing.pytorch as mox 
    moxing.file.shift('os', 'mox')
    print('------------------------')
    # mox.file.copy_parallel(cloud_prefix+'/apex-master/', '/cache/apex-master')
    mox.file.copy_parallel(cloud_prefix+'/apex-master.tar.gz', '/cache/apex-master.tar.gz')
    os.system('tar xvzf /cache/apex-master.tar.gz -C /cache/') 
    if os.system('pip --default-timeout=100 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /cache/apex-master')!=0:
        os.system('pip --default-timeout=100 install -v --no-cache-dir /cache/apex-master')
    import apex  

if use2243=='2243':    
    os.system('pip install easydict -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
    os.system('pip install cvbase -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
    os.system('pip install colorlog -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
    os.system('pip install gpustat -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
    os.system('pip install setuptools-scm -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
    os.system('pip install bcolz -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
    os.system('pip install tensorboardX -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
    os.system('pip install mxnet -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
    os.system('pip install ninja -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
    os.system('pip install msgpack-numpy -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
    os.system('pip install torchcontrib -i http://192.168.5.34:8888/repository/pypi/simple/ --trusted-host=192.168.5.34')
elif use2243=='6944':
    os.system('pip install easydict ')
    os.system('pip install cvbase ')
    os.system('pip install colorlog ')
    os.system('pip install gpustat ')
    os.system('pip install setuptools-scm ')
    os.system('pip install bcolz ')
    os.system('pip install tensorboardX ')
    os.system('pip install mxnet ')
    os.system('pip install ninja ')
    os.system('pip install msgpack-numpy ')
    os.system('pip install torchcontrib ')

from lz import *    
