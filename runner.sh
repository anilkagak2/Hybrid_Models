

#data='/projectnb/datascigrp/anilkag/Imagenet-1000/data'
data='/mnt/disks/data-disk/datasets/imagenet-1000' 

CUDA='0,1,2,3'
nGPUs=4

base='tf_mobilenetv3_small_100'
base_type='timm'

global='mobilenetv3_large_100'
global_type='timm'

CUDA_VISIBLE_DEVICES=$CUDA ./distributed_train.sh $nGPUs $data\
       --model $base --model-type $base_type --pretrained \
       --global-model $global --global-type $global_type --min-lr 1e-6\
       --lr 0.0004 --warmup-epochs 5 --epochs 25 --weight-decay 1e-5 --sched cosine \
       --reprob 0.2 --remode pixel --aa rand-m9-mstd0.5 -b 256 -j 12 \
       --model-ema --model-ema-decay 0.999 --opt adam --warmup-lr 1e-6 \
       --s_iters 1300 --t_iters 1300 --g_iters 610 --cov 0.45 --g_denom 0.4 \
       --eval-metric hybrid_acc \
       --amp --use-multi-epochs-loader --routing-model no_ft --n_parts 2 --routing-ema-decay 0.999



