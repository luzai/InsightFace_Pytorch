#!/usr/bin/env bash

for w1 in .1 # .3 .5. .7 .9
do
    for w2 in .1 #.3 .5 .7 .9
    do
        echo '!!' w1 w2 >> res.log
        source activate torch
        python comb.py --w1 "$w1" --w2 "$w2"
        conda deactivate
        ./run.sh >> res.log
    done
done