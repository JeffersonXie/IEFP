#!/bin/bash  


date

CUDA_VISIBLE_DEVICES=0,1 python test.py  --batch-size 512 --multiprocessing-distributed --world-size 1 --rank 0 

date