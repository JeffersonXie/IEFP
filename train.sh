#!/bin/bash  


date


CUDA_VISIBLE_DEVICES=0,1 python train.py --lambda1 0.001 --batch-size 512 --fp16 True --spu_scale 1 \
        --multiprocessing-distributed --world-size 1 --rank 0 --lr 0.1 --schedule 10 18 22 --epochs 25

date