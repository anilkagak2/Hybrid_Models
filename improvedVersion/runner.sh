
#data='/home/anilkag/datasets/imagenet-1000/'
data='/Data/anil_datasets/imagenet-1000/'
#data='/mnt/active/datasets/imagenet/'

#student='vit_tiny_patch16_224'
#teacher='vit_large_patch16_224'

student='tf_mobilenetv3_small_100'
teacher='mobilenetv3_large_100'

epochs=90
batch_size=64

CUDA_VISIBLE_DEVICES='0,1,2,3' ./distributed_train.sh 4 $data -b $batch_size \
	--model $student --global_model $teacher -j 16 \
        --opt adamw  --warmup-lr 1e-6 \
	--sched cosine --epochs $epochs --lr 1e-4 --amp --weight-decay 5e-4 \
	--model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5-inc1 --pretrained


#	--model $student --teacher $teacher --routing $routing -j 16 \
