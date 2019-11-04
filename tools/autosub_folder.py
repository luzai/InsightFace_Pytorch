from modelarts.session import Session
from modelarts.estimator import Estimator

for _ in range(9):
    print('')
reps = ['modelarts.p3.large', 'modelarts.bm.gpu.8v100',
        'modelarts.vm.cpu.2u.1',  'modelarts.p3.4xlarge', 'modelarts.p3.2xlarge']

rep = reps[1]
region_name= 'cn-north-4'  

if region_name=='cn-north-4': 
    bucket = 'bucket-2243' 
else:
    bucket = 'bucket-6944' 
session = Session(w3_account='z84115054', w3_password='zl@201719', 
    app_id='ved_intern_formal', region_name=region_name)

import random
for codep in [
    'fcpth.r100',
    'fcpth.dop.3.27', 'fcpth.dop.6.27',
    'fcpth.neg.2.2.top3',
    'fcpth.neg.2.2.top3.dop',
    'fcpth.neg.25.1.top3',
    'fcpth.neg.48.02.top3',
    'fcpth.neg.48.02.top3.dop',
    'fcpth.r50',
    'fcpth.r50.bs',
    'fcpth.r50.ranger',
    'fcpth.r100.ranger',
    'fcpth.r100.ranger.1e-1',
]: # 'fcpth.r100.bs','fcpth.neg.25.1.top3.dop', 'fcpth.in'  'fcpth.ft'
    estimator = Estimator(
      modelarts_session=session,
      framework_type='PyTorch',                                     # AI引擎名称
      framework_version='PyTorch-1.1.0-python3.6',                  # AI引擎版本
      code_dir=f'/{bucket}/zl/{codep}/',       # 训练脚本目录
      boot_file=f'/{bucket}/zl/{codep}/train.py',            # 训练启动脚本目录 
      log_url=f'/{bucket}/zl/{codep}/',                    # 训练日志目录
      hyperparameters=[ dict(label='cutoff', value='10'),
          dict(label='work_path', value=f's3://{bucket}/zl/work_space/{codep}.ct10')
       ], 
      output_path=f'/{bucket}/zl/{codep}/',                                # 训练输出目录
      train_instance_type=rep,                  # 训练环境规格
      train_instance_count=1,                                       # 训练节点个数
      job_description=f'{codep}')       # 训练作业描述
    job_instance = estimator.fit(inputs=f'/{bucket}/zl/zl/datasets/alpha_f64/', 
        wait=False, job_name=f"{codep.replace('.','-')}-{random.randint(0,999)}")
    
