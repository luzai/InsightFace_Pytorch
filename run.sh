#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.13.72.84" --master_port=12345 ./train.py

#for wmdm in 1.5,1.0 1.0,1.0 1.1,1.86 1.2,1.56 1.3,1.33 1.4,1.15 # 1.0,2.25
#do
#    IFS=","; set -- $wmdm;
#    wm=$1; dm=$2; echo $wm and $dm
#    python train.py --mbfc_wm $wm --mbfc_dm $dm --work_path work_space/mbfc.$wm.$dm.retina.arc --epochs 1
#done

#for i in 4 3 2 1
#do
#    cp config.$i.py config.py
#    python train.py
#done

#for i in $(seq 64 -2 56)
#do
#    echo $i
#    python train.py --scale $i --epochs 1 --prof True --work_path work_space/bak.$i
#done

# 85*4=340
#python train.cotching.py --batch_size 340 --mutual_learning 0 --train_mode cotch --work_path work_space/mbfc.cotch.fx.acc_grad --acc_grad 2
#python train.cotching.py --batch_size 340 --mutual_learning 0.001 --train_mode mual --work_path work_space/mbfc.mual.1e-3 --acc_grad 2
python train.cotching.py --batch_size 340 --mutual_learning 0.001 --train_mode cotch --work_path work_space/mbfc.cotch.mual.1e-3.cont --acc_grad 2 --start_epoch 10 --start_step 157380

#python train.cotching.py --batch_size 340 --mutual_learning 1 --train_mode mual --work_path work_space/mbfc.mual.1.d256.new

#python train.py --batch_size 512 --acc_grad 2 --epochs 32 --work_path work_space/mbfc.d256.long
#python train.py --batch_size 512 --acc_grad 2 --scale 60 --work_path work_space/mbfc.d256.s60
#python train.py --batch_size 512 --acc_grad 2 --scale 58 --work_path work_space/mbfc.d256.s58
#python train.py --batch_size 512 --acc_grad 2 --loss arcfaceneg --margin .3 --margin2 .2 --work_path work_space/mbfc.d256.arcneg
#python train.py --batch_size 512 --acc_grad 2 --cutoff 5 --work_path work_space/mbfc.d256.c5
#python train.py --batch_size 512 --acc_grad 2 --cutoff 10 --work_path work_space/mbfc.d256.c10
#python train.py --batch_size 512 --acc_grad 2 --net_mode effnet --input_size 224 --scale 32 --work_path work_space/effnet.d256
