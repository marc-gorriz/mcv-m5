#!/bin/bash

cp /home/master/datatmp/master/Experiments/TT100K_trafficSigns/vvg/weights.hdf5 /home/master/datatmp/master/Experiments/BelgiumTSC/vvg-transfer

cp /home/master/datatmp/master/Experiments/TT100K_trafficSigns/vvg/weights.hdf5 /home/master/datatmp/master/Experiments/KITTI/vvg-kitti-fine-tunin

echo "=========================="
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif-crop.py -e vvg-crop -s /data/module5 -l ~/datatmp/ &>vvg-crop.log

echo "=========================="
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif-preprocessing1.py -e vvg-preprocessing1 -s /data/module5 -l ~/datatmp/ &>log.vvg-preprocessing1

echo "=========================="
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif-preprocessing2.py -e vvg-preprocessing2 -s /data/module5 -l ~/datatmp/ &>log.vvg-preprocessing2


echo "=========================="
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif-preprocessing3.py -e vvg-preprocessing3 -s /data/module5 -l ~/datatmp/ &>log.vvg-preprocessing3


echo "=========================="
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif-transfer-learning.py -e vvg-transfer -s /data/module5 -l ~/datatmp/ &>log.vvg-transfer


echo "=========================="
CUDA_VISIBLE_DEVICES=0 python train.py -c config/kitti_classif-from-scratch.py -e vvg-kitti-scratch -s /data/module5 -l ~/datatmp/ &>log.vvg-kitti-scratch


echo "=========================="
CUDA_VISIBLE_DEVICES=0 python train.py -c config/kitti_classif-fine-tuning.py -e vvg-kitti-fine-tuning -s /data/module5 -l ~/datatmp/ &>log.vvg-kitti-fine-tuning

echo "=========================="
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif-ResNet-from-scratch.py -e ResNet-scratch -s /data/module5 -l ~/datatmp/ &>log.ResNet-scratch

echo "=========================="
CUDA_VISIBLE_DEVICES=0 python train.py -c config/kitti_classif-fine-tuning.py -e vvg-kitti-fine-tuning -s /data/module5 -l ~/datatmp/ &>log.ResNet-fine-tuning
