
#data='/home/anilkag/datasets/imagenet-1000/'
data='/Data/anil_datasets/imagenet-1000/'
#data='/mnt/active/datasets/imagenet/'

#student='vit_tiny_patch16_224'
#teacher='vit_large_patch16_224'

#student='tf_mobilenetv3_small_100'
#teacher='mobilenetv3_large_100'
#batch_size=512

student='vit_tiny_patch16_224'
teacher='vit_large_patch16_224'

disk_router='DiSK_Router'
hybrid_router='Hybrid_Router'

epochs=100
batch_size=256 #512
ckpt='./output/train/20230331-192414-tf_mobilenetv3_small_100-224/last.pth.tar'

CUDA_VISIBLE_DEVICES='0,1,2,3' ./distributed_train.sh 4 $data -b $batch_size \
	--model $student --global_model $teacher -j 16 \
        --disk_router $disk_router --hybrid_router $hybrid_router \
        --opt adamw  --warmup-lr 1e-6 \
	--sched cosine --epochs $epochs --lr 5e-4 --amp --weight-decay 5e-4 \
	--model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5-inc1 \
        --pretrained #--resume $ckpt #--torchcompile

#	--sched cosine --epochs $epochs --lr 1e-4 --amp --weight-decay 5e-4 \
# torchcompile is supported only on GPU with capabilities >= 7.0

#	--model $student --teacher $teacher --routing $routing -j 16 \
