#!/usr/bin/env bash
ROOT=/home/xinglu/prj/
FROM=$ROOT/InsightFace_Pytorch
NAME=ZJU-ArtificialIdiot-FaceRecognition.2
DST=$ROOT/$NAME
MODEL=emore.r152.cont

rm $DST/* -rf
rsync -avzxXP $FROM/ $DST/ --exclude='work_space/'
rm $DST/.git $DST/.idea -rf
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
