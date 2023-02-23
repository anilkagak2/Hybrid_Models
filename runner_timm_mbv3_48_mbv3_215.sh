

#data='/projectnb/datascigrp/anilkag/Imagenet-1000/data'
data='/mnt/disks/data-disk/datasets/imagenet-1000' 

#CUDA_VISIBLE_DEVICES='0,1,2,3' ./distributed_train.sh 4 $data\
#      -b 128 --model tf_mobilenetv3_small_100 --pretrained \
#      --sched cosine --epochs 10 --lr 0.0001 --amp --remode pixel \
#      --reprob 0.2 --aa rand-m9-mstd0.5 --drop 0.2 --drop-connect 0.2 

# --dist-bn reduce

CUDA='0,1,2,3'
CUDA_VISIBLE_DEVICES=$CUDA ./distributed_train.sh 4 $data\
       --model 'tf_mobilenetv3_small_100' --model-type 'timm' --pretrained \
       --global-model 'mobilenetv3_large_100' --global-type 'timm' --min-lr 1e-6\
       --lr 0.0004 --warmup-epochs 5 --epochs 25 --weight-decay 1e-5 --sched cosine \
       --reprob 0.2 --remode pixel --aa rand-m9-mstd0.5 -b 256 -j 12 \
       --model-ema --model-ema-decay 0.999 --opt adam --warmup-lr 1e-6 \
       --s_iters 1300 --t_iters 1300 --g_iters 610 --cov 0.45 --g_denom 0.4 \
       --eval-metric hybrid_acc \
       --amp --use-multi-epochs-loader --routing-model no_ft --n_parts 2 --routing-ema-decay 0.999


#       --model 'tf_mobilenetv3_small_100' --model-type 'timm' --pretrained \
#       --global-model 'mobilenetv3_large_100' --global-type 'timm' --min-lr 1e-6\
#       --global-model 'flops@595M_top1@80.0_finetune@75' --global-type 'ofa_spec' --min-lr 1e-6\
#       --model 'note8_lat@22ms_top1@70.4_finetune@25' --model-type 'ofa_spec' --pretrained \
#       --model 'mcunet-5fps_imagenet' --model-type 'mcunet' --pretrained \

#CUDA_VISIBLE_DEVICES='0,1,2,3' ./distributed_train.sh 4 $data\
#       --model tf_mobilenetv3_small_100 --model-type timm --pretrained \
#       --global-model mobilenetv3_large_100 --global-type timm --min-lr 1e-6\
#       --lr 0.0002 --warmup-epochs 5 --epochs 50 --weight-decay 1e-5 --sched cosine \
#       --reprob 0.2 --remode pixel --aa rand-m9-mstd0.5 -b 256 -j 6 \
#       --model-ema --model-ema-decay 0.9999 --opt adam --warmup-lr 1e-6 \
#       --s_iters 1300 --t_iters 1300 --g_iters 260 --cov 0.35 --g_denom 2.0 \
#       --eval-metric hybrid_acc \
#       --amp --use-multi-epochs-loader --routing-model no_ft --n_parts 2 --routing-ema-decay 0.999

#       --s_iters 1010 --t_iters 1010 --g_iters 210 --cov 0.35 --g_denom 2.0 \
#       --amp --use-multi-epochs-loader --routing-model no_ft --n_parts 1 --routing-ema-decay 0.99       
#       --amp --dist-bn reduce --use-multi-epochs-loader

