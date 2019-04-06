#!/usr/bin/env bash

#set -x

DEVKIT="/data2/share/megaface/devkit/experiments"
ALGO="insightface"
ROOT='/data2/share/megaface/results'

FEALOC="emore.r152.ada.chkpnt.3"
RESNAME="$FEALOC.cl.res"

python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model "work_space/$FEALOC/save" --output "$ROOT/$FEALOC"

python -u remove_noises.py --algo "$ALGO" --feature_dir_input "$ROOT/$FEALOC" --feature_dir_out "$ROOT/$FEALOC.cl"

echo $ROOT
cd "$DEVKIT"
source activate py2
rm -rf ../../$RESNAME
LD_LIBRARY_PATH="/data/xinglu/anaconda3/envs/py2/lib:$LD_LIBRARY_PATH" python2 -u run_experiment.py "$ROOT/$FEALOC/megaface" "$ROOT/$FEALOC/facescrub" _"$ALGO".bin ../../$RESNAME/ -s 1000000,100000 -p ../templatelists/facescrub_features_list.json
#LD_LIBRARY_PATH="/data/xinglu/anaconda3/envs/py2/lib:$LD_LIBRARY_PATH" python2 -u run_experiment.py "$ROOT/$FEALOC/megaface" "$ROOT/$FEALOC/facescrub" _"$ALGO".bin ../../$RESNAME/ -s 100000 -p ../templatelists/facescrub_features_list_10000.4.json

conda deactivate
cd -

