#!/usr/bin/env bash

# 'mbv3.lrg.5.casia' 'mbv3.lrg.5.casia.hd' 'mbv3.lrg5.retina' 'mbv3.lrg.5.casia.arc'
#for mp in 'mbv3.lrg.5.casia.sft' 'mbv3.lrg.5.casia.cos.32' 'mbv3.lrg.5.casia.cos.64'

for mp in 'mbv3.lrg.5.casia.arc' 'mbv3.lrg.5.casia.cos.64'
do
    echo $mp
#    python test_validate_ori.py --modelp $mp
    python test_ijbc3.py --modelp $mp
done
