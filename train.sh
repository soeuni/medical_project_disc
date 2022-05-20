python train.py --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3 --n-epochs 30 --image-size 256

python train.py --kernel-type 5fold_b5_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b5  --n-epochs 30 --image-size 256

python train.py --kernel-type 10fold_b3_512_30ep --k-fold 10 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30 --image-size 512 --batch-size 4

python train.py --kernel-type 10fold_b4_512_50ep --k-fold 10 --data-folder original_stone/ --enet-type tf_efficientnet_b4_ns --n-epochs 50 --image-size 512 --batch-size 4

python train.py --kernel-type 10fold_b4_512_30ep --k-fold 10 --data-folder original_stone/ --enet-type tf_efficientnet_b4_ns --n-epochs 30 --image-size 512 --batch-size 4

python train.py --kernel-type 10fold_b4_512_100ep --k-fold 10 --data-folder original_stone/ --enet-type tf_efficientnet_b4_ns --n-epochs 100 --image-size 512 --batch-size 4



