#!/usr/bin/env bash
sudo mkdir -p /media/mem_data
sudo mount -t tmpfs -o size=40g  tmpfs /media/mem_data
cd /media/mem_data
mkdir -p faces_ms1m_112x112 faces_casia ms1m-retinaface-t1
#cp /data1/share/faces_ms1m_112x112/train.* faces_ms1m_112x112/
#cp /data1/share/faces_casia/train.* faces_casia/
cp /data1/share/ms1m-retinaface-t1/train.rec ms1m-retinaface-t1/
cp /data1/share/ms1m-retinaface-t1/train.idx ms1m-retinaface-t1/

rsync amax86:/data1/share/faces_ms1m_112x112/train.tc.idx faces_ms1m_112x112/
rsync amax86:/data1/share/faces_casia/train.tc.idx faces_casia/
rsync amax86:/data1/share/ms1m-retinaface-t1/train.tc.idx ms1m-retinaface-t1/
