#!/usr/bin/env bash
export PYTHONPATH=.
python tools/clean.py &
python train.py --epochs 5 --work_path work_space/ft.ipabn --batch_size 768 --ipabn True --cvt_ipabn False 
# python train.py --epochs 5 --work_path work_space/ft.ipabn.neg.2.2 --batch_size 768 --ipabn True --cvt_ipabn False --loss arcfaceneg --margin .2 --margin2 .2 --topk 3 

# python train.py --epochs 16 --work_path work_space/r50.elu.casia.arcft.in --batch_size 1024 --acc_grad 1 --loss 'arcface' --margin .5 --ds False --embedding_size 512 --arch_ft True --use_in True

#CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.13.72.84" --master_port=12345 ./train.py

#for wmdm in 1.5,1.0 1.0,1.0 1.1,1.86 1.2,1.56 1.3,1.33 1.4,1.15 # 1.0,2.25
#do
#    IFS=","; set -- $wmdm;
#    wm=$1; dm=$2; echo $wm and $dm
#    python train.py --mbfc_wm $wm --mbfc_dm $dm --work_path work_space/mbfc.$wm.$dm.retina.arc --epochs 1
#done

# 85*4=340
#python train.cotching.py --batch_size 340 --mutual_learning 0 --train_mode cotch --work_path work_space/mbfc.cotch.fx.acc_grad --acc_grad 2
#python train.cotching.py --batch_size 340 --mutual_learning 0.001 --train_mode mual --work_path work_space/mbfc.mual.1e-3 --acc_grad 2
#python train.cotching.py --batch_size 340 --mutual_learning 0.001 --train_mode cotch --work_path work_space/mbfc.cotch.mual.1e-3 --acc_grad 2

#python train.py --batch_size 512 --acc_grad 2 --epochs 32 --work_path work_space/mbfc.d256.long
#python train.py --batch_size 512 --acc_grad 2 --scale 60 --work_path work_space/mbfc.d256.s60
#python train.py --batch_size 512 --acc_grad 2 --scale 58 --work_path work_space/mbfc.d256.s58
#python train.py --batch_size 512 --acc_grad 2 --loss arcfaceneg --margin .3 --margin2 .2 --work_path work_space/mbfc.d256.arcneg
#python train.py --batch_size 512 --acc_grad 2 --cutoff 5 --work_path work_space/mbfc.d256.c5
#python train.py --batch_size 512 --acc_grad 2 --cutoff 10 --work_path work_space/mbfc.d256.c10
#python train.py --batch_size 512 --acc_grad 2 --net_mode effnet --input_size 224 --scale 32 --work_path work_space/effnet.d256

#python train.py --scale 32 --net_mode mobilefacenet --work_path work_space/mbfc2 --epochs 38 --fill_cache 0

#python train.py --scale 32 --net_mode sglpth --lambda_runtime_reg 5 --work_path work_space/sglpth3 --epochs 38
#python train.2.py --scale 32  --net_mode sglpth --conv2dmask_drop_ratio 0 --work_path work_space/sglpth3.2.32 --epochs 38

#python train.2.py --scale 48 --net_mode sglpth  --conv2dmask_drop_ratio 0 --work_path work_space/sglpth2.2.48
#python train.3.py --scale 48 --net_mode sglpth  --conv2dmask_drop_ratio 0 --work_path work_space/sglpth2.3.48

#python train.py --epochs 16 --work_path work_space/r18.ep16
#python train.cotching.py --batch_size 256 --mutual_learning 0 --train_mode cotch --work_path work_space/r18.cotch.tau.1.38 --acc_grad 4 --tau 0.1 --epochs 38

#for mualt in 64 48 100 10 32
#do
#    python train.cotching.py --batch_size 256 --mutual_learning $mualt --train_mode mual --work_path work_space/r18.mual.again3.$mualt --acc_grad 4 --tau 0 --epochs 16
#done

#for times in 1
#do
#    python train.py --epochs 18 --work_path work_space/mbfc.se.prelu.ms1m.$times --batch_size 400 --acc_grad 3 --fill_cache .3
#done

#for times in  2 3
#do
#    python train.py --epochs 18 --work_path work_space/mbfc.se.prelu.ms1m.$times --batch_size 400 --acc_grad 3 --fill_cache .5
#done

#for times in 1
#do
#    python train.py --epochs 76 --work_path work_space/mbfc.se.prelu.adamarcface.ep76.$times --batch_size 400 --acc_grad 3 --fill_cache .7
#done


#for times in 1
#do
#for scale in 48 # 64
#do
#for betas1 in 0.95 0.9
#do
#for n_sma in 5 4
#do
#for lr in 0.003 0.001
#do
#    python train.py --epochs 18 --work_path work_space/n1.irse.elu.casia.ranger.lr$lr.$scale.$betas1.$n_sma.specnrm.$times --batch_size 512 --acc_grad 2 --loss 'arcface' --lr $lr --use_opt 'ranger'  --weight_decay 0.00004 --scale $scale --adam_betas1 $betas1 --n_sma $n_sma --spec_norm True --margin .5
#done
#done
#done
#done
#done

# for times in 1
# do
# for scale in 48
# do
# for betas1 in 0.95
# do
# for n_sma in 4
# do
#     python train.py --epochs 18 --work_path work_space/n1.irse.elu.casia.ranger.lr3e-3.$scale.$betas1.$n_sma.in.$times --batch_size 512 --acc_grad 2 --loss 'arcface' --lr 0.003 --use_opt 'ranger' --weight_decay 0.00004 --scale $scale --adam_betas1 $betas1 --n_sma $n_sma --spec_norm True --margin .5 --use_in True
# done
# done
# done
# done

#for times in 1 2 3
#do
#python train.py --epochs 18 --work_path work_space/irse.elu.casia.arc.ft.$times --batch_size 512 --acc_grad 2 --loss 'arcface' --margin .5 --ds False --embedding_size 512 --arch_ft True
#done

#python train.py --epochs 18 --work_path work_space/irse.elu.casia.arc --batch_size 512 --acc_grad 2 --loss 'arcface' --margin .5 --ds False --embedding_size 512 --arch_ft False

#python train.py --epochs 18 --work_path work_space/irse.elu.casia.arc.ft --batch_size 512 --acc_grad 2 --loss 'arcface' --margin .5 --ds False --embedding_size 512 --arch_ft True
#
#python train.py --epochs 18 --work_path work_space/irse.elu.casia.arcft.spcnrmmore --batch_size 512 --acc_grad 2 --loss 'arcface' --margin .5 --ds False --embedding_size 512 --arch_ft True --spec_norm True


#python train.py --epochs 18 --work_path work_space/irse.elu.casia.arcneg.t2 --batch_size 500 --acc_grad 2 --loss 'arcfaceneg' --margin .35 --margin2 .1 --topk 10 --ds False --spec_norm True

#python train.py --epochs 18 --work_path work_space/irse.elu.casia.arc.mid.bl --batch_size 480 --acc_grad 2 --loss 'arcface' --margin .5 --mid_type gpool --ds False

#python train.py --epochs 18 --work_path work_space/irse.elu.casia.arc.mid.bl.ds --batch_size 440 --acc_grad 2 --loss 'arcface' --margin .5 --ds True --mid_type gpool

#for topk in 5
#do
#for scale in 48
#do
#for wmdm in 0.48,0.02 0.35,0.15 0.4,0.1 0.45,0.05 0.3,0.1
#do
#IFS=","; set -- $wmdm;
#margin=$1; margin2=$2; echo $margin and $margin2
#python train.py --epochs 18 --work_path work_space/n2.irse.elu.casia.arcneg.$topk.$margin.$margin2.specnrm --batch_size 512 --acc_grad 2 --loss 'arcfaceneg' --scale $scale --margin $margin --margin2 $margin2  --topk $topk --ds False --spec_norm True
#done
#done
#done


