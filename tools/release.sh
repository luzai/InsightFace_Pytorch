#!/usr/bin/env bash

set -e

ROOT=/home/xinglu/prj/
FROM=$ROOT/InsightFace_Pytorch
NAME=ZJU-ArtificialIdiot-FaceRecognition
DST=$ROOT/$NAME
MODEL=asia.emore.r50.5

rm $DST/* -rf
rsync -avzxXP $FROM/ $DST/ --exclude='work_space/'
rm $DST/.git $DST/.idea log* __pycache__ -rf
#python tmp.py

mkdir -pv $DST/work_space/
rsync -avzxXP $FROM/work_space/$MODEL $DST/work_space/ --exclude='models/'
pushd $DST/work_space/$MODEL
ln -sf ./save ./models
cd save
rm optimizer* head* -rf
popd

cd $ROOT
rm $NAME.zip
zip -r $NAME.zip $NAME
