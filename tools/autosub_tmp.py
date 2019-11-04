import pandas as pd
from modelarts.session import Session
from modelarts.estimator import Estimator
import random

def parse_spec(spec):
    res = []
    for kv in spec.strip(';').split(';'):
        k,v = kv.split('=')
        res.append(dict(label=k, value=v)) 
    return res 

for _ in range(9):
    print('')

reps = ['modelarts.p3.large', 'modelarts.bm.gpu.8v100',
        'modelarts.vm.cpu.2u.1',  'modelarts.p3.4xlarge', 'modelarts.p3.2xlarge']

rep = reps[1]
region_name= 'cn-north-4' # 'cn-north-4' #
 
if region_name=='cn-north-4': 
    bucket='bucket-2243' 
else:
    bucket= 'bucket-6944' 
session = Session(w3_account='z84115054', 
    # w3_password='zl@201719', 
    app_id='ved_intern_formal', 
    app_token='4e9e1c53-c4d9-4882-9d5a-1df7c0349ce7' , 
    region_name=region_name)

from modelarts.session import Session
from modelarts.estimator import Estimator
resp = Estimator.get_job_list(
    modelarts_session=session, per_page = 1000,
    )
df = pd .DataFrame(resp['jobs'])
for idx, row in df.iterrows():
    if row.status == '11':
        estimator = Estimator(session, job_id=str(row.job_id))
        job_version_instance_list = estimator.get_job_version_object_list(is_show=False)
        if len(job_version_instance_list)==1:
            print(row.job_desc)
            Estimator.delete_job_by_id(modelarts_session=session, job_id = str(row.job_id))

# estimator = Estimator(modelarts_session=session, job_id="32155", version_id="110899")
# resp = estimator.get_job_log_file_list()
# resp = estimator.get_job_log(log_file='job-mg5-nochsfst-256.0')
# print(resp)

# estimator = Estimator(session, job_id="job89f8852b")
# job_version_instance_list = estimator.get_job_version_object_list()
# print(job_version_instance_list)