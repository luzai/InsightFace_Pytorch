#!/usr/bin/env bash
sudo mkdir -p /media/mem_data
sudo mount -t tmpfs -o size=20g  tmpfs /media/mem_data
cd /media/mem_data
mkdir -p faces_ms1m_112x112 faces_casia
cp /data1/share/faces_ms1m_112x112/train.* faces_ms1m_112x112/
cp /data1/share/faces_casia/train.* faces_casia/
rsync amax86:/data1/share/faces_ms1m_112x112/train.tc.idx faces_ms1m_112x112/
rsync amax86:/data1/share/faces_casia/train.tc.idx faces_casia/
