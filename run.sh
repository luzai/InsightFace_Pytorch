#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.13.72.84" --master_port=12345 ./train.py

for wmdm in 1.5,1.0 1.0,1.0 1.1,1.86 1.2,1.56 1.3,1.33 1.4,1.15 # 1.0,2.25
do
    IFS=","; set -- $wmdm;
    wm=$1; dm=$2; echo $wm and $dm
    python train.py --mbfc_wm $wm --mbfc_dm $dm --work_path work_space/mbfc.$wm.$dm.retina.arc --epochs 1
done
