#!/usr/bin/env bash

set -e

DEVKIT="/data2/share/megaface/devkit/experiments"
ROOT='/data2/share/megaface/results'
CODE_ROOT='/home/xinglu/prj/InsightFace_Pytorch'
ALGO="insightface"


FEALOC="hrnet.retina.arc.3"
RESNAME="$FEALOC.cl.res"

#python -u gen_megaface.py --gpu 1 --algo "$ALGO" --model "$CODE_ROOT/work_space/$FEALOC/save" --output "$ROOT/$FEALOC"
#
#python -u remove_noises.py --algo "$ALGO" --feature_dir_input "$ROOT/$FEALOC" --feature_dir_out "$ROOT/$FEALOC.cl"

cd "$DEVKIT"
source activate py2
rm -rf "$ROOT/$RESNAME"

LD_LIBRARY_PATH="/data/xinglu/anaconda3/envs/py2/lib:$LD_LIBRARY_PATH" python2 -u run_experiment.py "$ROOT/$FEALOC.cl/megaface" "$ROOT/$FEALOC.cl/facescrub" _"$ALGO".bin "$ROOT/$RESNAME" -p ../templatelists/facescrub_features_list.json -s 1000000

#LD_LIBRARY_PATH="/data/xinglu/anaconda3/envs/py2/lib:$LD_LIBRARY_PATH" python2 -u run_experiment.py "$ROOT/$FEALOC.cl/megaface" "$ROOT/$FEALOC.cl/facescrub" _"$ALGO".bin "$ROOT/$RESNAME" -s 100000 -p ../templatelists/facescrub_features_list_10000.4.json

conda deactivate
cd -

