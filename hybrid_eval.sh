#DATA='/mnt/active/datasets/imagenet' 
#DATA='/mnt/disks/data-disk/datasets/imagenet-1000' 
#DATA='/home/anilkag/datasets/imagenet-1000/'
#DATA='/projectnb/datascigrp/anilkag/datasets/imagenet-1000/' #../../../../../
DATA='/projectnb/datascigrp/anilkag/Imagenet-1000/data'


#resume_ckpt='./models/ofa-ofa_150M_constrained_base-ofa-ofa_150M_constrained_global-no_ft-trn-2-checkpoint.pth.tar'
resume_ckpt='./models/ofa-ofa_150M_constrained_base-ofa-ofa_150M_constrained_global-no_ft-trn-2-model_best.pth.tar'

workers=8
batch_size=512 #256 #512 #1024

n_parts=1
#base_type='timm' 
#base_arch='tf_mobilenetv3_small_100' 
#base_arch='tf_mobilenetv3_large_075' 
#base_arch='mobilenetv3_large_100'
base_type='ofa_spec' #'mcunet' #'ofa_spec'
base_arch='note8_lat@22ms_top1@70.4_finetune@25'
#base_type='ofa'
#base_arch='f2_ofa_150M_constrained_base'
#base_arch='f_ofa_150M_constrained_base'
#base_arch='re4_ofa_150M_constrained_base'
#base_arch='re3_ofa_150M_constrained_base'
#base_arch='re_ofa_250M_constrained_base'
#base_arch='re2_ofa_250M_constrained_base'
#base_arch='re3_ofa_250M_constrained_base'
#base_arch='re2_ofa_150M_constrained_base'
#base_arch='ofa_150M_constrained_base'
#base_arch='ofa_250M_constrained_base'
#base_arch='ofa_350M_constrained_base'
#base_arch='re_ofa_350M_constrained_base'
#base_arch='re2_ofa_350M_constrained_base'
#base_type='mcunet'
#base_arch='mcunet-5fps_imagenet'

#global_type='timm' 
#global_arch='mobilenetv3_large_100'
global_type='ofa_spec' 
global_arch='flops@595M_top1@80.0_finetune@75' 
#global_type='ofa' 
#global_arch='f2_ofa_150M_constrained_global'
#global_arch='f_ofa_150M_constrained_global'
#global_arch='re4_ofa_150M_constrained_global'
#global_arch='re3_ofa_150M_constrained_global'
#global_arch='re_ofa_250M_constrained_global'
#global_arch='re2_ofa_250M_constrained_global'
#global_arch='re3_ofa_250M_constrained_global'
#global_arch='mcunet_base_fixed_global_search'
#global_arch='re2_ofa_150M_constrained_global'
#global_arch='ofa_150M_constrained_global'
#global_arch='ofa_250M_constrained_global'
#global_arch='ofa_350M_constrained_global'
#global_arch='re_ofa_350M_constrained_global'
#global_arch='re2_ofa_350M_constrained_global'

routing_arch='no_ft'
#routing_arch='with_ft'

#resume_ckpt="./models/${base_type}-${base_arch}-${global_type}-${global_arch}-${routing_arch}-trn-${n_parts}-checkpoint.pth.tar"
#resume_ckpt='./output/train/20211016-181405-tf_mobilenetv3_small_100-224/last.pth.tar'
#resume_ckpt='./output/train/20211016-181405-tf_mobilenetv3_small_100-224/model_best.pth.tar'
#resume_ckpt='./output/train/20211016-181405-tf_mobilenetv3_small_100-224/checkpoint-15.pth.tar'
#resume_ckpt='./output/train/20211018-095352-timm-tf_mobilenetv3_small_100-timm-mobilenetv3_large_100-no_ft-trn-2--224/checkpoint-14.pth.tar'
#resume_ckpt='./output/train/20211018-133324-timm-tf_mobilenetv3_small_100-timm-mobilenetv3_large_100-no_ft-trn-2--224/checkpoint-6.pth.tar'
#resume_ckpt='./output/train/20211018-181247-timm-tf_mobilenetv3_small_100-timm-mobilenetv3_large_100-no_ft-trn-2--224/last.pth.tar'
#resume_ckpt='./output/train/20211020-133513-ofa_spec-note8_lat@22ms_top1@70.4_finetune@25-ofa_spec-flops@595M_top1@80.0_finetune@75-with_ft-trn-1--140/last.pth.tar'
resume_ckpt='./output/train/20211020-165629-ofa_spec-note8_lat@22ms_top1@70.4_finetune@25-ofa_spec-flops@595M_top1@80.0_finetune@75-no_ft-trn-1--140/checkpoint-52.pth.tar'
log_file="./final_logs/2080Ti-eval-${base_type}-${base_arch}-${global_type}-${global_arch}-${routing_arch}-${n_parts}.txt"
echo "${log_file}"
echo "${resume_ckpt}"
CUDA='0,1,2' #'0,1,2,3'
CUDA_VISIBLE_DEVICES=$CUDA python  eval_hybrid.py --batch-size $batch_size -j $workers \
        --base_type $base_type --base_arch $base_arch \
        --global_type $global_type --global_arch $global_arch \
        --routing_arch $routing_arch --n_parts $n_parts \
	--train-dir $DATA/train --val-dir $DATA/val --path $DATA --eval --load_from $resume_ckpt  #| tee -a $log_file
#        --resume --load_from $resume_ckpt \
#	--epochs $epochs --wd $wd --resume --load_from $resume_ckpt | tee -a $log_file



