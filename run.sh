#!/usr/bin/bash

##########################################################
#     Pseudo Label 2013
##########################################################
##=== ePseudoLabel2013v1 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=64 --usp-batch-size=64 --num-labels=4000 --arch=cnn13 --model=epslab2013v1 --usp-weight=1.0 --soft=False --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=True --save-freq=100 2>&1 | tee results/epslab2013v1_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== ePseudoLabel2013v2 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=False --num-labels=4000 --arch=cnn13 --model=epslab2013v2 --usp-weight=1.0 --soft=False --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=True --save-freq=100 2>&1 | tee results/epslab2013v2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== iPseudoLabel2013v1 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=64 --usp-batch-size=64 --num-labels=4000 --arch=cnn13 --model=ipslab2013v1 --usp-weight=1.0 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=False --save-freq=100 2>&1 | tee results/ipslab2013v1_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== iPseudoLabel2013v2 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=False --num-labels=4000 --arch=cnn13 --model=ipslab2013v2 --usp-weight=1.0 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=False --save-freq=100 2>&1 | tee results/ipslab2013v2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##########################################################
#     MixUp Pseudo Label 2013
##########################################################
##=== eMixPseudoLabelv1 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=64 --usp-batch-size=64 --num-labels=4000 --arch=cnn13 --model=epslab2013v1 --usp-weight=1.0 --soft=False --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=True --save-freq=100 2>&1 | tee results/epslab2013v1_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== eMixPseudoLabelv2 ===
# cifar10-4k
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=False --num-labels=4000 --arch=cnn13 --model=emixpslabv2 --usp-weight=30. --mixup-alpha=1.0 --soft=True --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=True --save-freq=100 2>&1 | tee results/emixpslabv2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt


##########################################################
#     Tempens 2017
##########################################################
##=== eTempensv1 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=64 --usp-batch-size=64 --num-labels=4000 --arch=cnn13 --model=etempensv1 --usp-weight=30.0 --ema-decay=0.6 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=True --save-freq=100 2>&1 | tee results/etempensv1_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== eTempensv2 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=False --num-labels=4000 --arch=cnn13 --model=etempensv2 --usp-weight=30.0 --ema-decay=0.6 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=True --save-freq=100 2>&1 | tee results/etempensv2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== iTempensv1 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=64 --usp-batch-size=64 --num-labels=4000 --arch=cnn13 --model=itempensv1 --usp-weight=30.0 --ema-decay=0.6 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=True --save-freq=100 2>&1 | tee results/itempensv1_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== iTempensv2 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=False --num-labels=4000 --arch=cnn13 --model=itempensv2 --usp-weight=30.0 --ema-decay=0.6 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=True --save-freq=100 2>&1 | tee results/itempensv2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt


##########################################################
#     PI 2017
##########################################################
##=== PIv1 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=64 --usp-batch-size=64 --num-labels=4000 --arch=cnn13 --model=piv1 --usp-weight=30.0 --drop-ratio=0.5 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=exp-warmup --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-twice=True --save-freq=100 2>&1 | tee results/piv1_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== PIv2 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=False --num-labels=4000 --arch=cnn13 --model=piv2 --usp-weight=30.0 --drop-ratio=0.5 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=exp-warmup --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-twice=True --save-freq=100 2>&1 | tee results/piv2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt


##########################################################
#     MeanTeacher 2017
##########################################################
##=== MeanTeacherv1 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=64 --usp-batch-size=64 --num-labels=4000 --arch=cnn13 --model=mtv1 --usp-weight=30.0 --ema-decay=0.97 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-twice=True --save-freq=100 2>&1 | tee results/mtv1_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== MeanTeacherv2 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=False --num-labels=4000 --arch=cnn13 --model=mtv2 --usp-weight=30.0 --ema-decay=0.97 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-twice=True --save-freq=100 2>&1 | tee results/mtv2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt


##########################################################
#     ICT 2019
##########################################################
##=== ICTv1 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=64 --usp-batch-size=64 --num-labels=4000 --arch=cnn13 --model=ictv1 --usp-weight=30.0 --mixup-alpha=1.0 --ema-decay=0.97 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-twice=True --save-freq=100 2>&1 | tee results/ictv1_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== ICTv2 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=False --num-labels=4000 --arch=cnn13 --model=ictv2 --usp-weight=30.0 --mixup-alpha=1.0 --ema-decay=0.97 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-twice=False --save-freq=100 2>&1 | tee results/ictv2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt


##########################################################
#     MixMatch 2019
##########################################################
##=== MixMatchv1 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=True --num-labels=4000 --arch=cnn13 --model=mixmatch --usp-weight=30.0 --mixup-alpha=1.0 --ema-decay=0.97 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-twice=True --save-freq=100 2>&1 | tee results/mixmatchv1_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

##=== MixMatchv2 ===
# cifar10-4k
#CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=False --num-labels=4000 --arch=cnn13 --model=mixmatch --usp-weight=30.0 --mixup-alpha=1.0 --ema-decay=0.97 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-twice=True --save-freq=0 2>&1 | tee results/mixmatchv2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt
