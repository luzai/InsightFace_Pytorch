from modelarts.session import Session
from modelarts.estimator import Estimator
import random

dry_run=True#False#
def parse_spec(*args):
    args = [t.strip(';') for t in args]
    spec = ';'.join(args)
    res = []
    for kv in spec.strip(';').split(';'):
        k, v = kv.split('=')
        res.append(dict(label=k, value=v))
    return res

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

spec_gl = "epochs=5;lr=1e-3;acc_grad=1;head_init=all;margin=.4;ipabn=False;cvt_ipabn=True;head_load=False;schedule=mstep;"  # head_parallel=data;ipabn=True;cvt_ipabn=False;cutoff=10;load_from=s3://bucket-2243/zl/work_space/emore.r100.dop.alphaf64.chsfst
spec_lcs = [
    "tri_wei=.9;",
    # "tri_wei=.9;cutoff=10;",
    # "tri_wei=.9;instances=8;lr=1e-4",
    # "tri_wei=.9;instances=16;",
    # "tri_wei=.9;cutoff=10;tri_scale=1;",
    # "tri_wei=.9;cutoff=10;tri_scale=1;pos_wei=2;neg_wei=50;",
    # "tri_wei=.9;cutoff=10;tri_scale=1;pos_wei=2;neg_wei=40;",
    # "tri_wei=.9;tri_scale=64;pos_wei=.5;neg_wei=2;instances=8;mining=dop;mining_init=-1;rand_ratio=.333",
    # "tri_wei=.9;tri_scale=64;pos_wei=.5;neg_wei=2;instances=8;mining=dop;mining_init=-1;rand_ratio=.222",
    # "tri_wei=.9;tri_scale=64;pos_wei=.5;neg_wei=4;instances=8;mining=dop;mining_init=-1;rand_ratio=.333",
    # "tri_wei=.9;cutoff=10;tri_scale=1;pos_wei=2;neg_wei=50;mining=dop;mining_init=-1;rand_ratio=.162;",

    # "margin=.4;schedule=cycle;epochs=9;cycle_up=0.;cycle_down=1.;",
    # "margin=.4;schedule=cycle;epochs=9;cycle_up=0.;cycle_down=1.;lr=1e-4",
    # "margin=.4;schedule=cycle;epochs=9;cycle_up=0.;cycle_down=1.;lr=5e-4",
    # "margin=.4;schedule=cycle;epochs=9;cycle_up=.3;cycle_down=.7;",
    # "margin=.4;schedule=cycle;epochs=9;cycle_up=.0;cycle_down=1.;swa=origin;cycle_momentum=True;lr_mult=10;",
    # "margin=.4;schedule=cycle;epochs=9;cycle_up=.0;cycle_down=1.;swa=modify;cycle_momentum=True;lr_mult=10;",
    # "margin=.4;schedule=cycle;epochs=9;cycle_up=0.;cycle_down=1.;cycle_gamma=.9",
    # "margin=.4;schedule=cycle;epochs=9;cycle_up=0.;cycle_down=1.;cycle_gamma=1.",
    # "margin=.4;schedule=cycle;epochs=9;cycle_up=.0;cycle_down=1.;cycle_gamma=1.;swa=origin",
    # "margin=.4;schedule=cycle;epochs=9;cycle_up=0.;cycle_down=1.;cycle_momentum=True;lr_mult=10;",

    # "margin=.5;",
    # "margin=.4;cutoff=10;",
    # "margin=.4;cutoff=10;ipabn=True;cvt_ipabn=False;",
    # "margin=.4;cutoff=10;ipabn=True;cvt_ipabn=False;head_parallel=data;",
    # "margin=.45;",
    # "margin=.4;head_init=None;head_load=True",
    # "margin=.4;head_init=10",
    # "margin=.4;head_init=all;head_init_zjjk=True",
    # "margin=.4;head_init=all;head_init_zjjk=True;lr_mult=10;lr=1e-4;",

    # "margin=.4;weight_decay=5e-4;",
    # "margin=.4;weight_decay=5e-4;wd_mult=1;",
    # "margin=.4;weight_decay=5e-4;wd_mult=1;lr_mult=10;",
    # "margin=.4;lr=5e-4;",
    # "margin=.4;lr=1e-4;",
    # "mining=dop;mining_init=-1;rand_ratio=.777;",
    # "mining=dop;mining_init=-1;rand_ratio=.111;",
    # "mining=dop;mining_init=-1;rand_ratio=.333;",
    # "loss=arcfaceneg;margin=.25;margin2=.15;topk=3;",
    # "loss=arcfaceneg;margin=.25;margin2=.15;topk=3;mining=dop;mining_init=-1;rand_ratio=.162;",
    # "loss=arcfaceneg;margin=.25;margin2=.15;topk=3;mining=dop;mining_init=-1;rand_ratio=.333;pos_wei=.5;neg_wei=2;instances=8;",
    # "loss=arcfaceneg;margin=.48;margin2=.02;topk=3;",
    # "loss=arcfaceneg;margin=.48;margin2=.02;topk=3;mining=dop;mining_init=-1;rand_ratio=.162;",
    # "loss=arcfaceneg;margin=.3;margin2=.1;topk=10;",
    # "loss=arcfaceneg;margin=.2;margin2=.2;topk=10;",
    # "loss=arcfaceneg;margin=.2;margin2=.2;topk=10;mining=dop;mining_init=-1;rand_ratio=.333;pos_wei=.5;neg_wei=2;instances=8;lr_mult=10",
    # "loss=arcfaceneg;margin=.25;margin2=.1;topk=3;mining=dop;mining_init=-1;rand_ratio=.162;tri_wei=.1;",
]
names = [
    'tri9',
    # 'tri9.ct10',
    # 'tri9.inst8.decay10',
    # 'tri9.inst16',
    # 'tri9.ct10.tris1', 'tri9.ct10.tris1.p2.n50',
    # 'tri9.ct10.tris1.p2.n40',
    # 'tri9.p.5.n2.inst8.dop333',
    # 'tri9.p.5.n2.inst8.dop222',
    # 'tri9.p.5.n4.inst8.dop333',
    # 'tri9.ct10.tris1.p2.n50.dop',

    # 'mg4.cyc1',
    # 'mg4.cyc1.1en4',
    # 'mg4.cyc1.5en4',
    # 'mg4.cyc7',
    # 'mg4.cyc.swa.mom.lm10',
    # 'mg4.cyc.swa.mod.mom.lm10',
    # 'mg4.cyc1.gm9',
    # 'mg4.cyc1.gm1',
    # 'mg4.cyc1.gm1.swa',
    # 'mg4.cyc1.mom.lm10',

    # 'mg5',
    # 'mg4.ct10',
    # 'mg4.ct10.ipabn',
    # 'mg4.ct10.ipabn.headp',
    # 'mg45',
    # 'mg4.fromwei',
    # 'mg4.fromfea.10',
    # 'mg4.fromfea.zjjk.ld2',
    # 'mg4.fromfea.zjjk.lm10.ld2.decay10',

    # 'mg4.wd5en4',
    # 'mg4.wd5en4.wm1',
    # 'mg4.wd5en4.wm1.lm10',
    # 'mg4.decay2',
    # 'mg4.decay10',
    # 'mg4.dop.777',
    # 'mg4.dop.111',
    # 'mg4.dop.333',
    # 'neg.25.15',
    # 'neg.25.15.dop',
    # 'neg.25.15.dop333.tri',
    # 'neg.48.02',
    # 'neg.48.02.dop',
    # 'neg.3.1.top10',
    # 'neg.2.2.top10',
    # 'neg.2.2.top10.dop333.tri.lm10',
    # 'neg.25.1.dop.tri',
]
names = ['ft4.' + n + '' for n in names]
print('run ', len(names))
for spec_lc, name in zip(spec_lcs, names):
    print(name, '\t', spec_lc, )
assert len(names) == len(spec_lcs)
codep = 'fcpth'

for spec_lc, name in zip(spec_lcs, names):
    spec_nm = f"work_path=s3://{bucket}/zl/work_space/{name};"
    clean_name = f"{name.replace('.','-')}-{random.randint(100,999)}"

    print('python train.py ', end=' ')
    for kv in parse_spec(spec_gl, spec_lc, spec_nm):
        v,k= kv['value'],kv['label']
        print(f'--{k} {v}', end=' ')
    print()

    if not dry_run:
        estimator = Estimator(
            modelarts_session=session,
            framework_type='PyTorch',  # AI引擎名称
            framework_version='PyTorch-1.1.0-python3.6',  # AI引擎版本
            code_dir=f'/{bucket}/zl/{codep}/',  # 训练脚本目录
            boot_file=f'/{bucket}/zl/{codep}/train.py',  # 训练启动脚本目录
            log_url=f'/{bucket}/zl/{codep}/{clean_name}/',  # 训练日志目录
            hyperparameters=parse_spec(spec_gl, spec_lc, spec_nm),
            output_path=f'/{bucket}/zl/{codep}/{clean_name}/',  # 训练输出目录
            train_instance_type=rep,  # 训练环境规格
            train_instance_count=1,  # 训练节点个数
            job_description=clean_name)  # 训练作业描述
        job_instance = estimator.fit(inputs=f'/{bucket}/zl/zl/datasets/alpha_f64/',
                                     wait=False, job_name=clean_name)
