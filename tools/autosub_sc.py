from modelarts.session import Session
from modelarts.estimator import Estimator
import random


def parse_spec(spec):
    res = []
    for kv in spec.strip(';').split(';'):
        k, v = kv.split('=')
        res.append(dict(label=k, value=v))
    return res


for _ in range(9):
    print('')

reps = ['modelarts.p3.large', 'modelarts.bm.gpu.8v100',
        'modelarts.vm.cpu.2u.1', 'modelarts.p3.4xlarge', 'modelarts.p3.2xlarge']

rep = reps[1]
region_name = 'cn-north-4'  # 'cn-north-4' #

if region_name == 'cn-north-4':
    bucket = 'bucket-2243'
else:
    bucket = 'bucket-6944'
session = Session(w3_account='z84115054',
                  # w3_password='zl@201719',
                  app_id='ved_intern_formal', app_token='4e9e1c53-c4d9-4882-9d5a-1df7c0349ce7', region_name=region_name)

spec_gl = "epochs=18;ipabn=False;cvt_ipabn=False;lr=1e-1;acc_grad=1;cutoff=0;milestones=9_13;chs_first=False;load_from=None;"  #head_parallel=data;
spec_lcs = [
    # "margin=.4;",
    "margin=.5;",
    # "margin=.4;tri_wei=.1;",
    # "margin=.4;tri_wei=.5;",
    # "margin=.4;tri_wei=.9;",
    # "margin=.4;cutoff=5;",
    # "margin=.4;cutoff=15;",
    # "mining=dop;mining_init=-1;rand_ratio=.162;",
    # "loss=arcfaceneg;margin=.25;margin2=.1;topk=3;",
    # "loss=arcfaceneg;margin=.25;margin2=.1;topk=3;mining=dop;mining_init=-1;rand_ratio=.162;",
    # "mining=dop;mining_init=-1;rand_ratio=.111;",
    # "mining=dop;mining_init=-1;rand_ratio=.333;",
    # "loss=arcfaceneg;margin=.48;margin2=.02;topk=3;",
    # "loss=arcfaceneg;margin=.48;margin2=.02;topk=3;mining=dop;mining_init=-1;rand_ratio=.162;",
    # "loss=arcfaceneg;margin=.25;margin2=.25;topk=10;",
    # "loss=arcfaceneg;margin=.3;margin2=.2;topk=10;",
    # "loss=arcfaceneg;margin=.3;margin2=.1;topk=10;",
    # "loss=arcfaceneg;margin=.2;margin2=.2;topk=10;",
    # "loss=arcfaceneg;margin=.25;margin2=.1;topk=3;mining=dop;mining_init=-1;rand_ratio=.162;tri_wei=.1;",
]
names = [
    # 'mg4.nochsfst',
    'mg5.nochsfst',
    # 'mg4.tri.1',
    # 'mg4.tri.5',
    # 'mg4.tri.9',
    # 'mg4.ct5',
    # 'mg4.ct15',
    # 'dop.162',
    # 'neg.25.1',
    # 'neg.25.1.dop',
    # 'dop.111',
    # 'dop.333',
    # 'neg.48.02',
    # 'neg.48.02.dop',
    # 'neg.25.25.top10',
    # 'neg.3.2.top10',
    # 'neg.3.1.top10',
    # 'neg.2.2.top10',
    # 'neg.25.1.dop.tri',
]
names = [ n for n in names]
assert len(names) == len(spec_lcs)
codep = 'fcpth'
for spec_lc, name in zip(spec_lcs, names):
    spec_nm = f"work_path=s3://{bucket}/zl/work_space/{name};"
    clean_name = f"{name.replace('.','-')}-{random.randint(100,999)}"
    estimator = Estimator(
        modelarts_session=session,
        framework_type='PyTorch',  # AI引擎名称
        framework_version='PyTorch-1.1.0-python3.6',  # AI引擎版本
        code_dir=f'/{bucket}/zl/{codep}/',  # 训练脚本目录
        boot_file=f'/{bucket}/zl/{codep}/train.py',  # 训练启动脚本目录
        log_url=f'/{bucket}/zl/{codep}/{clean_name}/',  # 训练日志目录
        hyperparameters=parse_spec(spec_gl + spec_lc + spec_nm),
        output_path=f'/{bucket}/zl/{codep}/{clean_name}/',  # 训练输出目录
        train_instance_type=rep,  # 训练环境规格
        train_instance_count=1,  # 训练节点个数
        job_description=clean_name)  # 训练作业描述
    print(parse_spec(spec_gl + spec_lc + spec_nm))
    job_instance = estimator.fit(inputs=f'/{bucket}/zl/zl/datasets/alpha_f64/',
                                 wait=False, job_name=clean_name)
    # break
