#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.13.72.84" --master_port=12345 ./train.py

#for cfg in config.lrg5.py #config.sml.py config.sml5.py
#do
#    echo $cfg
#    cp $cfg config.py
#    python train.py
#done
#cp config.bk.py config.py
python tools/test_ijbc3.py --modelp casia.r20.arc
python train.py
python tools/test_ijbc3.py --modelp casia.r50.arc.scrth

